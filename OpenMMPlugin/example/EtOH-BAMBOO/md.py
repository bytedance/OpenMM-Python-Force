import ase.io
from rdkit import Chem
import numpy as np
import torch

from openmm import CMMotionRemover, HarmonicBondForce, LangevinMiddleIntegrator, Platform, System, VerletIntegrator
from openmm.app import Element, Simulation, Topology, PDBFile, PDBReporter, StateDataReporter
import openmm.unit as unit

from Wrapper import WrapperGasPhase
from CallbackPyForce import Callable
from CallbackPyForce import TorchForce as CbkTorchForce
from openmmtorch import TorchForce as OmmTorchForce


def get_bamboo_torchforce(ckpt, device, atom_types, mode):
    wrapper = WrapperGasPhase(ckpt=ckpt, device=device, atom_types=atom_types)
    if "compile" in mode:
        wrapper = torch.compile(wrapper)
    elif "script" in mode or "omm" in mode:
        wrapper = torch.jit.script(wrapper)

    if "omm" in mode:
        torchforce = OmmTorchForce(wrapper)
        torchforce.setUsesPeriodicBoundaryConditions(False)
        torchforce.setOutputsForces(False)
        call =  "dummy"
    else:
        call = Callable(id(wrapper), Callable.RETURN_ENERGY)
        torchforce = CbkTorchForce(call)

    if "graph" in mode:
        torchforce.setProperty("useCUDAGraphs", "true")

    return wrapper, call, torchforce  # must return all of them to keep them alive


def get_atom_types(sdf: str):
    types = []
    rd_supplier = Chem.SDMolSupplier(sdf, removeHs=False)
    rd_mol = rd_supplier[0]
    for rd_a in rd_mol.GetAtoms():
        types.append(rd_a.GetAtomicNum())
    return types


def get_single_molecule_openmm_simulation(sdf, dt, T, random_seed, friction, forces, device):
    osys = System()
    ase_atoms = ase.io.read(sdf, index=0)
    for ase_a in ase_atoms:
        osys.addParticle(ase_a.mass)
    pos_nm = ase_atoms.positions * 0.1

    obond = HarmonicBondForce()
    otop = Topology()
    ochain = otop.addChain()
    oresidue = otop.addResidue("MOL", ochain)

    rd_supplier = Chem.SDMolSupplier(sdf, removeHs=False)
    rd_mol = rd_supplier[0]
    rd_mapping = dict()
    for rd_a in rd_mol.GetAtoms():
        symbol = rd_a.GetSymbol()
        o_a = otop.addAtom(name=symbol, element=Element.getBySymbol(symbol), residue=oresidue)
        rd_mapping[rd_a.GetIdx()] = o_a
    for rd_b in rd_mol.GetBonds():
        at1, at2 = rd_mapping[rd_b.GetBeginAtomIdx()], rd_mapping[rd_b.GetEndAtomIdx()]
        otop.addBond(at1, at2)
        obond.addBond(rd_b.GetBeginAtomIdx(), rd_b.GetEndAtomIdx(), 0.1, 0.0)  # 0.1 nm, 0.0 kJ/mol
    osys.addForce(obond)
    for force in forces:
        osys.addForce(force)
    osys.addForce(CMMotionRemover())

    if device.startswith("cuda") or device.startswith("CUDA"):
        oplf = Platform.getPlatformByName("CUDA")
        oplf.setPropertyDefaultValue("Precision", "mixed")
    else:
        oplf = Platform.getPlatformByName("Reference")

    if friction == 0.0:
        oitg = VerletIntegrator(dt)
    else:
        oitg = LangevinMiddleIntegrator(T, friction, dt)
        oitg.setRandomNumberSeed(random_seed)

    osim = Simulation(otop, osys, oitg, oplf)
    osim.context.setPositions(pos_nm)
    osim.context.setVelocitiesToTemperature(T, random_seed)
    return osim


def analyze_potential_energy(osim: Simulation, pos_nm):
    osim.context.setPositions(pos_nm)
    st = osim.context.getState(getForces=True, getEnergy=True)
    e = st.getPotentialEnergy()._value
    f = [[ff.x, ff.y, ff.z] for ff in st.getForces()]
    return e, f


def analyze_potential_energies(osim: Simulation, ifilename: str, ofilename: str):
    e_all, f_all = [], []
    ase_atoms = ase.io.read(ifilename, index=":")
    for ase_a in ase_atoms:
        if ase_a:
            pos_A = ase_a.positions * unit.angstrom
            pos_nm = pos_A / unit.nanometer
            e, f = analyze_potential_energy(osim, pos_nm)
            e_all.append(e)
            f_all.append(f)
    if ofilename is None:
        print(np.array(e_all))
        print(np.array(f_all))
    else:
        np.savez(ofilename, e=np.array(e_all), f=np.array(f_all))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--job", type=str, choices=["simulate", "analyze", "minimize"], help="job")
    # model
    ap.add_argument("--ckpt", type=str, default="bamboo/benchmark/benchmark.pt", help="checkpoint path")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda"], help="can only be cuda")
    ap.add_argument("--mode", type=str, default="omm", help="other values: native, script, and compile; optional suffix: '.graph'")
    # analyze
    ap.add_argument("--dt", type=float, default=0.001, help="time-step; unit ps")
    ap.add_argument("--ifile", nargs="+", help="ifile(s)")
    ap.add_argument("--ofile", nargs="+", help="ofile(s)")
    ap.add_argument("--random-seed", type=int, default=1, help="random seed")
    ap.add_argument("--T", type=float, default=300., help="T; unit Kelvin")
    ap.add_argument("--friction", type=float, default=0.1, help="friction; unit 1/ps")
    # simulate
    ap.add_argument("--nsave", type=int, default=1, help="save 1 trajectory frame every X md steps")
    ap.add_argument("--nframe", type=int, default=0, help="save X trajectory frames")
    args = ap.parse_args()
    print(args)

    def main_analyze(args: argparse.Namespace):
        sdf = args.ifile[0]
        coordfile = args.ifile[1]
        ofilename = None
        if args.ofile is None:
            ofilename = None
        else:
            ofilename = args.ofile[0]

        atom_types = get_atom_types(sdf)
        wrapper, call, torchforce = get_bamboo_torchforce(args.ckpt, args.device, atom_types, args.mode)
        osim = get_single_molecule_openmm_simulation(sdf, args.dt, args.T, args.random_seed, 0.0, [torchforce], args.device)
        analyze_potential_energies(osim, coordfile, ofilename)

    def main_simulate(args: argparse.Namespace):
        sdf = args.ifile[0]
        ofilestem = args.ofile[0]
        pdbfile = f"{ofilestem}.pdb"
        csvfile = f"{ofilestem}.csv"

        atom_types = get_atom_types(sdf)
        wrapper, call, torchforce = get_bamboo_torchforce(args.ckpt, args.device, atom_types, args.mode)
        osim = get_single_molecule_openmm_simulation(sdf, args.dt, args.T, args.random_seed, args.friction, [torchforce], args.device)

        nsave = args.nsave
        nsteps = nsave * args.nframe

        if args.job == "simulate" and nsteps > 0:
            coordfile = args.ifile[1]
            ase_atoms = ase.io.read(coordfile, index=0)
            pos_nm = ase_atoms.positions * 0.1
            osim.context.setPositions(pos_nm)
            coord_reporter = PDBReporter(pdbfile, nsave)
            state_reporter = StateDataReporter(open(csvfile, "w"), nsave, step=True, time=True,
                                               potentialEnergy=True, kineticEnergy=True,
                                               temperature=True, elapsedTime=True, separator=",")
            osim.reporters.append(coord_reporter)
            osim.reporters.append(state_reporter)
            osim.step(nsteps)
        elif args.job == "minimize":
            osim.minimizeEnergy()
            minpos = osim.context.getState(getPositions=True).getPositions()
            with open(pdbfile, "w") as fmin:
                PDBFile.writeFile(osim.topology, minpos, fmin)

    if args.job == "analyze":
        main_analyze(args)
    elif args.job == "simulate" or args.job == "minimize":
        main_simulate(args)
