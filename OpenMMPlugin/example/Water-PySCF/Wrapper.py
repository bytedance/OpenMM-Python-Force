# contributor: Xiaojie WU (xiaojie.wu@bytedance.com)

import numpy as np
from numpy.typing import NDArray

from pyscf import gto
from pyscf.dft import rks as cpu_rks
try:
    from gpu4pyscf.dft import rks as gpu_rks
except:
    pass


class PySCFWrapper:

    def __init__(self, calc_setup, atom_types: list[int], charge: int, initial_pos: NDArray):
        pos_angstrom = initial_pos * 10

        mol = gto.Mole()
        mol.atom = [[itype, pos] for itype, pos in zip(atom_types, pos_angstrom)]

        mol.basis = calc_setup["basis"]
        mol.charge = charge
        mol.unit = "A"
        mol.build()
        self.mol = mol

        device = calc_setup["device"]
        if device == "cpu":
            mf = cpu_rks.RKS(mol, xc=calc_setup["xc"]).density_fit()
        elif device == "gpu":
            mf = gpu_rks.RKS(mol, xc=calc_setup["xc"]).density_fit()
        mf.disp = calc_setup["disp"]
        mf.grids.atom_grid = calc_setup["grids"]
        mf.kernel()
        self.calculator = mf.nuc_grad_method().as_scanner()
        self.calculator_e = mf.as_scanner()

    def __call__(self, pos: NDArray, *, includeForces: bool = True) -> tuple[float, NDArray]:
        """
        pos: shape(NAtoms, 3); unit: nm.
        includeForces: True: calculate energy and gradients; False: calculate energy and numpy.zeros_like(pos).
        returns: tuple[energy, gradient]; units: kJ/mol, kJ/mol/nm.
        """

        pos_angstrom = pos * 10

        mol = self.mol
        mol.set_geom_(pos_angstrom, unit="A")

        # Hartree, Hartree/Bohr -> kJ/mol, kJ/mol/nm
        if includeForces:
            e, g = self.calculator(mol)
            g_omm = g * 49613.7
        else:
            e = self.calculator_e(mol)
            g_omm = np.zeros_like(pos)

        e_omm = e * 2625.5
        return e_omm, g_omm
