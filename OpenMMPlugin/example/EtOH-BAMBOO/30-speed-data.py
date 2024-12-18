import numpy as np


def mainfunc():

    baseline_speed = None
    print("{:6s}{:>8s}{:>8s}{:>6s}".format("No.", "ms/step", "Msteps", "Rel."))
    for i in range(1, 9):
        stem = f"d{i}"
        csv = f"{stem}.csv"
        data = np.loadtxt(csv, comments=["#"], delimiter=",")

        nsteps = data[-1, 0]
        seconds = data[-1, -1]

        ms_per_step = 1000 * seconds / nsteps
        million_steps_per_day = 24 * 60 * 60 * 1000 / ms_per_step / 1.e6
        if i == 1:
            baseline_speed = million_steps_per_day
        relative_speed = million_steps_per_day / baseline_speed

        print(f"{stem:6s}{ms_per_step:8.3g}{million_steps_per_day:8.3g}{relative_speed:6.2g}")


if __name__ == "__main__":
    mainfunc()
