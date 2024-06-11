import subprocess
import numpy as np
import os
import argparse
import json_numpy

parser = argparse.ArgumentParser(
    prog="explore_many_showers",
    description="Simulate multiple showers and plot them.",
)
parser.add_argument(
    "out_dir",
    metavar="OUT_DIR",
    type=str,
    help=("The of output directory."),
)
parser.add_argument(
    "--particle",
    metavar="TYPE",
    type=str,
    default="gamma",
    help=("The primary particle type."),
)
parser.add_argument(
    "--energy_GeV",
    metavar="GEV",
    type=float,
    default=250.0,
    help=("Energy of the primary particle."),
)

args = parser.parse_args()


out_dir = "2024-60-11_run"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "config.json"), "wt") as f:
    f.write(
        json_numpy.dumps(
            {"particle": args.particle, "energy_GeV": args.energy_GeV}
        )
    )

for random_seed in range(1, 7):
    shower_dir = os.path.join(out_dir, "{:06d}".format(random_seed))
    if not os.path.exists(shower_dir):
        subprocess.call(
            [
                "python",
                "./corsika_simulation.py",
                shower_dir,
                "--particle",
                args.particle,
                "--random_seed",
                str(random_seed),
                "--energy_GeV",
                str(args.energy_GeV),
                "--num_observation_levels",
                str(15),
                "--cherenkov_histogram_radius",
                str(160),
            ]
        )

    shower_slides_dir = os.path.join(shower_dir, "slides")
    if not os.path.exists(shower_slides_dir):
        subprocess.call(
            [
                "python",
                "./plot_slides.py",
                os.path.join(shower_dir, "overview.json"),
                shower_slides_dir,
            ]
        )
