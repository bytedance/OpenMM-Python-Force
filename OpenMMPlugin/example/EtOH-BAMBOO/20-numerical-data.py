import numpy as np
import matplotlib.pyplot as plt
import ase.io


def calc_rmsd(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    d0 = a0.ndim
    d1 = a1.ndim
    assert d0 == d1
    if d0 == 2:
        return np.sqrt(np.mean((a1 - a0)**2, axis=1))
    assert False


def get_data(csv: str):
    return np.loadtxt(csv, comments=["#"], delimiter=",")


def get_coords(pdb: str):
    coords = []
    ase_atoms = ase.io.read(pdb, index=":")
    for frame in ase_atoms:
        if frame:
            coords.append(frame.positions.reshape(1, -1))
    return np.array(coords).squeeze(1)


xytickfontsize=32
xylabelfontsize=32
legendfontsize=32
dpi = 300
nsteps = np.arange(1, 1 + 100)
xlabel = "NVE Steps"
plt.rcParams["figure.figsize"] = (16, 9)
plt.rcParams["lines.linewidth"] = 5
ext = "pdf"


def drift_and_divergence():
    baseline_U, baseline_K, baseline_H = None, None, None
    baseline_r = None
    for i in range(1, 9):
        stem = f"nve{i}"
        csv = f"{stem}.csv"
        pdb = f"{stem}.pdb"
        data = get_data(csv)

        coords = get_coords(pdb)
        potential = data[:, 2]
        kinetic = data[:, 3]
        hamiltonian = potential + kinetic
        h_avg = np.mean(hamiltonian)
        h_std = np.std(hamiltonian)
        print(f"{stem} {h_avg:.1f} {h_std:.2f}")

        # plot drift

        drift_png = f"drift{i}.{ext}"
        plt.plot(nsteps, hamiltonian, label=f"#{i}")
        plt.xlim(nsteps[0], nsteps[-1])
        plt.hlines(h_avg, nsteps[0], nsteps[-1], linestyles="--", colors="k")
        plt.xlabel(xlabel, fontsize=xylabelfontsize)
        plt.ylabel("Hamiltonian (kJ/mol)", fontsize=xylabelfontsize)
        plt.legend(fontsize=legendfontsize)
        plt.tick_params(axis="both", labelsize=xytickfontsize)
        plt.tight_layout()
        plt.savefig(drift_png, dpi=dpi)
        plt.close()

        # divergence

        if i == 1:
            baseline_U = potential
            baseline_K = kinetic
            baseline_H = hamiltonian
            baseline_r = coords
        else:
            div_png = f"div{i}.{ext}"
            du, dk, dh = np.abs(potential - baseline_U), np.abs(kinetic - baseline_K), np.abs(hamiltonian - baseline_H)
            plt.plot(nsteps, du, label=f"#{i} $\Delta$U")
            plt.plot(nsteps, dk, label=f"#{i} $\Delta$K")
            plt.plot(nsteps, dh, label=f"#{i} $\Delta$H")
            plt.xlim(nsteps[0], nsteps[-1])
            plt.xlabel(xlabel, fontsize=xylabelfontsize)
            plt.ylabel("Unsigned Difference (kJ/mol)", fontsize=xylabelfontsize)
            plt.legend(fontsize=legendfontsize)
            plt.tick_params(axis="both", labelsize=xytickfontsize)
            plt.tight_layout()
            plt.savefig(div_png, dpi=dpi)
            plt.close()

            dr = np.abs(coords - baseline_r)
            print(f"max(RMSD) of R {np.max(dr)}")


def rerun():
    baseline_e, baseline_f = None, None
    for i in range(1, 9):
        stem = f"rerun{i}"
        npz = f"{stem}.npz"
        data = np.load(npz)
        energy, force = data["e"], data["f"]
        nf, na, nd = force.shape
        force = force.reshape(nf, na * nd)

        if i == 1:
            baseline_e, baseline_f = energy, force
        else:
            du = np.abs(energy - baseline_e)
            df = calc_rmsd(force, baseline_f)
            npz_png = f"{stem}.{ext}"
            plt.plot(nsteps, du, label=f"#{i} $\Delta$U")
            plt.plot(nsteps, df, label=f"#{i} $\Delta$F")
            plt.xlim(nsteps[0], nsteps[-1])
            plt.xlabel(xlabel, fontsize=xylabelfontsize)
            plt.ylabel("Numerical Difference", fontsize=xylabelfontsize)
            plt.legend(fontsize=legendfontsize)
            plt.tick_params(axis="both", labelsize=xytickfontsize)
            plt.tight_layout()
            plt.savefig(npz_png, dpi=dpi)
            plt.close()


def mainfunc():
    drift_and_divergence()
    rerun()


if __name__ == "__main__":
    mainfunc()
