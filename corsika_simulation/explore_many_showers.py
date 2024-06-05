import subprocess
import numpy as np
import os

particle = "gamma"
energy_GeV = 250.0

out_dir = "showers"
os.makedirs(out_dir, exist_ok=True)
for random_seed in range(1, 5):
    shower_dir = os.path.join(out_dir, "{:06d}".format(random_seed))
    if not os.path.exists(shower_dir):
        subprocess.call(
            [
                "python",
                "./corsika_simulation.py",
                shower_dir,
                "--particle",
                particle,
                "--random_seed",
                str(random_seed),
                "--energy_GeV",
                str(energy_GeV),
                "--num_observation_levels",
                str(7),
            ]
        )
        subprocess.call(
            [
                "python",
                "./plot_slides.py",
                os.path.join(shower_dir, "overview.json"),
                os.path.join(shower_dir, "slides"),
            ]
        )
