import os
import torch

_dirname = os.path.dirname(__file__)
import sys
sys.path.append(_dirname + "/bamboo")
import utils.rejit as rejit


class WrapperGasPhase(torch.nn.Module):
    def __init__(self, ckpt, device, atom_types):
        super().__init__()
        natoms = len(atom_types)
        edges = [[], []]
        for i in range(natoms):
            for j in range(i+1, natoms):
                edges[0].append(i)
                edges[0].append(j)
                edges[1].append(j)
                edges[1].append(i)
        try:
            self._model = rejit.convert(ckpt, device)
        except RuntimeError as e:
            print(f"cannot rejit.convert; trying torch.jit.load(). Previous RuntimeError:\n{e}\n")
            m = torch.jit.load(ckpt, map_location=device)
            m.device = torch.device(device)
            self._model = m
        self._edges = torch.nn.Parameter(torch.tensor(edges, dtype=torch.long), requires_grad=False).to(device)
        self._atom_types = torch.nn.Parameter(torch.tensor(atom_types).to(torch.long), requires_grad=False).to(device)
        self._hack_cell = torch.nn.Parameter(torch.tensor(-1.), requires_grad=False).to(device)
        self._hack_g_ewald = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False).to(device)
        self._model.eval()

    def forward(self, pos: torch.Tensor):
        shift = (pos[self._edges[0]] - pos[self._edges[1]]) * 10.0  # nm to angstrom
        inputs = {
            "edge_index": self._edges,
            "edge_cell_shift": shift,
            "coul_edge_index": self._edges,
            "coul_edge_cell_shift": shift,
            "disp_edge_index": self._edges,
            "disp_edge_cell_shift": shift,
            "atom_types": self._atom_types,
            "cell": self._hack_cell,
            "g_ewald": self._hack_g_ewald
        }
        outputs = self._model.forward(inputs)
        e = outputs["pred_energy"].to(torch.float32) * 4.184  # kcal/mol to kJ/mol
        return e
