import argparse
import numpy as np

from Wrapper import PySCFWrapper
from CallbackPyForce import Callable, NumPyForce

from openmm import CMMotionRemover, HarmonicBondForce, CustomIntegrator, LangevinMiddleIntegrator, Platform, System
from openmm.app import Element, Topology, Simulation, PDBReporter, StateDataReporter

# Water
atom_symbols = ["O", "H", "H"]
atom_types = [8, 1, 1]
atom_masses = [15.999, 1.008, 1.008]
atom_init_pos_nm = np.array([
    [0.0000000000, -0.0000000000, 0.1174000000],
    [-0.7570000000, -0.0000000000, -0.4696000000],
    [0.7570000000, 0.0000000000, -0.4696000000]]) * 0.1
pyscf_setup = {
    "xc": "b3lyp",
    "grids": (75, 302),
    "basis": "def2-tzvpp",
    "disp": "d3bj"
}


def mainfunc(args: argparse.Namespace):
    device = args.device
    pyscf_setup["device"] = device


    osys = System()
    for m in atom_masses:
        osys.addParticle(m)

    # Force 1
    obond = HarmonicBondForce()
    otop = Topology()
    ochain = otop.addChain()
    oresidue = otop.addResidue("MOL", ochain)
    o_atoms = []
    for symbol in atom_symbols:
        o_a = otop.addAtom(name=symbol, element=Element.getBySymbol(symbol), residue=oresidue)
        o_atoms.append(o_a)
    otop.addBond(o_atoms[0], o_atoms[1])
    otop.addBond(o_atoms[0], o_atoms[2])
    obond.addBond(0, 1, 0.1, 0.0)  # 0.1 nm, 0.0 kJ/mol
    obond.addBond(0, 2, 0.1, 0.0)  # 0.1 nm, 0.0 kJ/mol
    osys.addForce(obond)

    # Force 2: NumPyForce
    wrapper = PySCFWrapper(pyscf_setup, atom_types, 0, atom_init_pos_nm)
    call = Callable(id(wrapper), Callable.RETURN_ENERGY_GRADIENT)
    call.add_kwarg("includeForces")
    npforce = NumPyForce(call)
    osys.addForce(npforce)

    # Force 3
    osys.addForce(CMMotionRemover())


    random_seed = args.random_seed
    T = 300
    dt = args.dt
    friction = args.friction
    pdbfile = "water.pdb"
    csvfile = "water.csv"
    nstep_save = args.nsave
    nframe = args.nframe
    nsteps = nstep_save * nframe


    if device == "cpu":
        oplf = Platform.getPlatformByName("CPU")
    elif device == "gpu":
        oplf = Platform.getPlatformByName("CUDA")
        oplf.setPropertyDefaultValue("Precision", "mixed")
    if friction == 0.0:
        oitg = CustomIntegrator(0.001)
        oitg.addComputePerDof("v", "v+0.5*dt*f/m")
        oitg.addComputePerDof("x", "x+dt*v")
        oitg.addComputePerDof("v", "v+0.5*dt*f/m")
    else:
       oitg = LangevinMiddleIntegrator(T, friction, dt)
       oitg.setRandomNumberSeed(random_seed)
    oitg = LangevinMiddleIntegrator(T, friction, dt)
    oitg.setRandomNumberSeed(random_seed)
    osim = Simulation(otop, osys, oitg, oplf)
    osim.context.setPositions(atom_init_pos_nm)
    osim.context.setVelocitiesToTemperature(T, random_seed)


    if nsteps > 0:
        coord_reporter = PDBReporter(pdbfile, nstep_save)
        state_reporter = StateDataReporter(open(csvfile, "w"), nstep_save, step=True, time=True,
                                           potentialEnergy=True, kineticEnergy=True,
                                           temperature=True, elapsedTime=True, separator=",")
        osim.reporters.append(coord_reporter)
        osim.reporters.append(state_reporter)
        osim.step(nsteps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--device", type=str, choices=["cpu", "gpu"], default="gpu", help="cpu, or gpu")
    ap.add_argument("--dt", type=float, default=0.001, help="time-step; unit ps")
    ap.add_argument("--friction", type=float, default=0.0, help="friction; unit 1/ps")
    ap.add_argument("--nsave", type=int, default=1, help="save 1 trajectory frame every X md steps")
    ap.add_argument("--nframe", type=int, default=100, help="save X trajectory frames")
    ap.add_argument("--random-seed", type=int, default=1, help="random seed")
    args = ap.parse_args()
    print(args)
    mainfunc(args)
