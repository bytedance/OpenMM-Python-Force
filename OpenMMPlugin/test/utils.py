import numpy as np
from numpy.typing import NDArray

import openmm
from openmm.app import Simulation, Topology


def getOpenMMSimulation(natoms: int, device: str, omm_forces: list[openmm.Force]):
    h_mass = 1.008
    dt_ps = 0.001

    omm_system = openmm.System()
    for i in range(natoms):
        omm_system.addParticle(h_mass)
    for f in omm_forces:
        omm_system.addForce(f)

    omm_topology = Topology()
    omm_intg = openmm.VerletIntegrator(dt_ps)

    if device.startswith("cuda"):
        omm_platform = openmm.Platform.getPlatformByName("CUDA")
        omm_platform.setPropertyDefaultValue("Precision", "mixed")
    else:
        omm_platform = openmm.Platform.getPlatformByName("Reference")

    omm_simulation = Simulation(omm_topology, omm_system, omm_intg, omm_platform)
    return omm_simulation


def getEnergyForceFromOpenMMContext(omm_context, pos_nm):
    omm_context.setPositions(pos_nm)
    states = omm_context.getState(getForces=True, getEnergy=True)
    e = states.getPotentialEnergy()._value
    f = np.array([[f.x, f.y, f.z] for f in states.getForces()])
    return e, f


class ModuleNumpyCpuOnly:
    def __init__(self, atom_ids, bs, ks):
        self.nbonds = len(ks)
        assert len(ks) == len(atom_ids)
        assert len(ks) == len(bs)
        self.atom_ids = np.array(atom_ids)
        self.bs = np.array(bs)
        self.ks = np.array(ks)

    def __call__(self, pos: NDArray) -> tuple[float, NDArray]:
        energy = 0.0
        grad = np.zeros(pos.shape)
        for ibond in range(self.nbonds):
            a0, a1 = self.atom_ids[ibond]
            b = self.bs[ibond]
            k = self.ks[ibond]
            r0 = pos[a0]
            r1 = pos[a1]
            r01 = r1 - r0
            dr2 = r01.dot(r01)
            dr = np.sqrt(dr2) - b
            energy += 0.5 * k * dr**2
            igrad = k * dr * r01 / np.linalg.norm(r01)
            grad[a0] -= igrad
            grad[a1] += igrad
        return energy, grad


try:
    import torch

    class ModuleTorchReference(torch.nn.Module):
        def __init__(self, atom_ids, bs, ks):
            super().__init__()
            self.nbonds = len(ks)
            assert self.nbonds == len(atom_ids)
            assert self.nbonds == len(bs)
            self.atom_ids = torch.nn.Parameter(torch.tensor(atom_ids), requires_grad=False)
            self.bs = torch.nn.Parameter(torch.tensor(bs), requires_grad=False)
            self.ks = torch.nn.Parameter(torch.tensor(ks), requires_grad=False)

        def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            energy = torch.tensor(0., device=pos.device, dtype=pos.dtype)
            grad = torch.zeros(pos.shape, device=pos.device, dtype=pos.dtype)
            for ibond in range(self.nbonds):
                a0, a1 = self.atom_ids[ibond]
                b = self.bs[ibond]
                k = self.ks[ibond]
                r0 = pos[a0]
                r1 = pos[a1]
                r01 = r1 - r0
                dr2 = r01.dot(r01)
                dr = torch.sqrt(dr2) - b
                energy += 0.5 * k * dr**2
                igrad = k * dr * r01 / r01.norm()
                grad[a0] -= igrad
                grad[a1] += igrad
            return energy, grad

    class ModuleTorchEnergy(torch.nn.Module):
        def __init__(self, atom_ids, bs, ks):
            super().__init__()
            self.nbonds = len(ks)
            assert len(ks) == len(atom_ids)
            assert len(ks) == len(bs)
            self.atom_ids = torch.nn.Parameter(torch.tensor(atom_ids), requires_grad=False)
            self.bs = torch.nn.Parameter(torch.tensor(bs), requires_grad=False)
            self.ks = torch.nn.Parameter(torch.tensor(ks), requires_grad=False)

        def forward(self, pos: torch.Tensor) -> torch.Tensor:
            dr = (pos[self.atom_ids[:, 1:]] - pos[self.atom_ids[:, :-1]]).squeeze(1)
            delta = torch.norm(dr, dim=1) - self.bs
            return 0.5 * torch.sum(self.ks * delta * delta)

    class ModuleTorchEnergyGradient(torch.nn.Module):
        def __init__(self, atom_ids, bs, ks):
            super().__init__()
            self.nbonds = len(ks)
            assert len(ks) == len(atom_ids)
            assert len(ks) == len(bs)
            self.atom_ids = torch.nn.Parameter(torch.tensor(atom_ids), requires_grad=False)
            self.bs = torch.nn.Parameter(torch.tensor(bs), requires_grad=False)
            self.ks = torch.nn.Parameter(torch.tensor(ks), requires_grad=False)

        def forward(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            dr = (pos[self.atom_ids[:, 1:]] - pos[self.atom_ids[:, :-1]]).squeeze(1)
            delta = torch.norm(dr, dim=1) - self.bs
            energy = 0.5 * torch.sum(self.ks * delta * delta)
            energy.backward()
            grad = pos.grad
            return energy, grad
except ImportError:
    pass
