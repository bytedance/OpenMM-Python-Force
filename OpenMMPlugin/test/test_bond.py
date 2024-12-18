import pytest
import numpy as np
from openmm import HarmonicBondForce
_device_list = []
_device_list.append("cpu")
try:
    import torch
    if torch.cuda.is_available():
        _device_list.append("cuda")
except ImportError:
    pass

from python.CallbackPyForce import Callable
from python.CallbackPyForce import NumPyForce
_has_torch_force = False
try:
    from python.CallbackPyForce import TorchForce
    from utils import ModuleTorchReference
    _has_torch_force = True
except ImportError:
    pass

import utils


_coords = [[0.00, 0.00, 0.00], [0.03, 0.04, 0.12]]
_bonds = [[0, 1]]
_b0 = [0.1]
_k0 = [30000.]

def getOpenMMHarmonicBondForce():
    f = HarmonicBondForce()
    for i, b, k in zip(_bonds, _b0, _k0):
        f.addBond(i[0], i[1], b, k)
    return f


@pytest.mark.parametrize("device", _device_list)
def test_bond(device):
    eps = 1e-6
    natoms = len(_coords)


    # openmm
    sim0 = utils.getOpenMMSimulation(natoms, device, [getOpenMMHarmonicBondForce()])
    e0, f0 = utils.getEnergyForceFromOpenMMContext(sim0.context, _coords)
    print(f"\n-- OpenMM --")


    # NumPyForce
    return_flag = Callable.RETURN_ENERGY_GRADIENT
    m90 = utils.ModuleNumpyCpuOnly(_bonds, _b0, _k0)
    call90 = Callable(id(m90), return_flag)
    f90 = NumPyForce(call90)
    sim90 = utils.getOpenMMSimulation(natoms, device, [f90])
    e90, f90 = utils.getEnergyForceFromOpenMMContext(sim90.context, _coords)
    assert np.isclose(e0, e90, atol=eps)
    assert np.allclose(f0, f90, atol=eps)
    print(f"\n-- ModuleNumpyCpuOnly return_flag: {return_flag} jit_script: {False} --")


    if _has_torch_force:
        # TorchForce
        for jit_script, return_flag, ModuleClass in zip([False, True, True],
                                                       [Callable.RETURN_ENERGY_GRADIENT, Callable.RETURN_ENERGY, Callable.RETURN_ENERGY_GRADIENT],
                                                       [utils.ModuleTorchReference, utils.ModuleTorchEnergy, utils.ModuleTorchEnergyGradient]):
            # native torch.nn.Module
            m10 = ModuleClass(_bonds, _b0, _k0)
            m10 = m10.to(device=device)
            m10.eval()
            call10 = Callable(id(m10), return_flag)
            f10 = TorchForce(call10)

            sim10 = utils.getOpenMMSimulation(natoms, device, [f10])
            e10, f10 = utils.getEnergyForceFromOpenMMContext(sim10.context, _coords)
            assert torch.allclose(torch.tensor(e0), torch.tensor(e10), atol=eps)
            assert torch.allclose(torch.tensor(f0), torch.tensor(f10), atol=eps)


            # torch.jit.script
            m20 = ModuleClass(_bonds, _b0, _k0)
            m20 = m20.to(device=device)
            m20.eval()
            if jit_script:
                m20 = torch.jit.script(m20)
            call20 = Callable(id(m20), return_flag)
            f20 = TorchForce(call20)

            sim20 = utils.getOpenMMSimulation(natoms, device, [f20])
            e20, f20 = utils.getEnergyForceFromOpenMMContext(sim20.context, _coords)
            assert torch.allclose(torch.tensor(e0), torch.tensor(e20), atol=eps)
            assert torch.allclose(torch.tensor(f0), torch.tensor(f20), atol=eps)


            # torch.compile
            m30 = ModuleClass(_bonds, _b0, _k0)
            m30 = m30.to(device=device)
            m30.eval()
            m30 = torch.compile(m30)
            call30 = Callable(id(m30), return_flag)
            f30 = TorchForce(call30)

            sim30 = utils.getOpenMMSimulation(natoms, device, [f30])
            e30, f30 = utils.getEnergyForceFromOpenMMContext(sim30.context, _coords)
            assert torch.allclose(torch.tensor(e0), torch.tensor(e30), atol=eps)
            assert torch.allclose(torch.tensor(f0), torch.tensor(f30), atol=eps)
            print(f"\n-- {ModuleClass.__name__} return_flag: {return_flag} jit_script: {jit_script} --")
