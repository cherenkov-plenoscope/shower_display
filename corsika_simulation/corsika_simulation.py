import corsika_primary
import toy_air_shower as tas
import numpy as np
import json_numpy
import os
import argparse

parser = argparse.ArgumentParser(
    prog="corsika_simulation",
    description="Export shower statistics for various altitudes.",
)

parser.add_argument(
    "out_dir",
    metavar="PATH",
    type=str,
    help=("The of output directory."),
)
parser.add_argument(
    "--particle",
    metavar="PARTICLE",
    type=str,
    help=("Type of primary particle"),
    required=False,
    default="gamma",
)
parser.add_argument(
    "--random_seed",
    metavar="RANDOM_SEED",
    type=int,
    help=("Random seed > 0"),
    required=False,
    default=1,
)
parser.add_argument(
    "--energy_GeV",
    metavar="ENERGY_GEV",
    type=float,
    help=("Energy of primary particle in GeV."),
    required=False,
    default=10.0,
)
parser.add_argument(
    "--num_observation_levels",
    metavar="NUM_LEVELS",
    type=int,
    help=("Number of horizontal slices."),
    required=False,
    default=6,
)
parser.add_argument(
    "--cherenkov_histogram_radius",
    metavar="RADIUS_M",
    type=int,
    help=("Radius of histogram to collect Cherenkov light. In m."),
    required=False,
    default=160,
)
args = parser.parse_args()

RANDOM_SEED = args.random_seed

i8 = np.int64
f8 = np.float64

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

namibia = {
    "earth_magnetic_field_x_muT": 12.5,
    "earth_magnetic_field_z_muT": -25.9,
    "corsika_atmosphere_id": 10,
}

observation_level_hess_asl_m = 1.8e3
range_in_atmosphere_m = 30e3

observation_levels_asl_m = np.geomspace(
    observation_level_hess_asl_m,
    observation_level_hess_asl_m + range_in_atmosphere_m,
    args.num_observation_levels,
)

HIST_RADIUS_M = args.cherenkov_histogram_radius
assert HIST_RADIUS_M > 0

XY_BIN_EDGES_M = np.linspace(
    start=-HIST_RADIUS_M, stop=HIST_RADIUS_M, num=(HIST_RADIUS_M) + 1
)

RUN_ID = RANDOM_SEED
PRIMARY = {}
PRIMARY["particle_id"] = f8(
    corsika_primary.particles.identification.PARTICLES[args.particle]
)
PRIMARY["energy_GeV"] = f8(args.energy_GeV)
PRIMARY["theta_rad"] = f8(0.0)
PRIMARY["phi_rad"] = f8(0.0)
PRIMARY["depth_g_per_cm2"] = f8(0.0)


for iobs in range(len(observation_levels_asl_m)):
    iobs_key = "{:06d}".format(iobs)

    run = {
        "run_id": i8(RUN_ID),
        "event_id_of_first_event": i8(1),
        "observation_level_asl_m": f8(observation_levels_asl_m[iobs]),
        "earth_magnetic_field_x_muT": f8(
            namibia["earth_magnetic_field_x_muT"]
        ),
        "earth_magnetic_field_z_muT": f8(
            namibia["earth_magnetic_field_z_muT"]
        ),
        "atmosphere_id": i8(namibia["corsika_atmosphere_id"]),
        "energy_range": {
            "start_GeV": f8(PRIMARY["energy_GeV"] * 0.99),
            "stop_GeV": f8(PRIMARY["energy_GeV"] * 1.01),
        },
        "random_seed": corsika_primary.random.seed.make_simple_seed(RUN_ID),
    }

    steering_dict = {"run": run, "primaries": [PRIMARY]}

    out_path = os.path.join(out_dir, iobs_key)
    if not os.path.exists(out_path + ".o"):
        corsika_primary.corsika_primary(
            steering_dict=steering_dict,
            cherenkov_output_path=out_path + ".cherenkov.tar",
            particle_output_path=out_path + ".particles.dat",
            stdout_path=out_path + ".o",
            stderr_path=out_path + ".e",
        )
        corsika_primary.particles.dat_to_tape(
            out_path + ".particles.dat", out_path + ".particles.tar"
        )


ID_TO_NAME = {}
NAME_TO_ID = corsika_primary.particles.identification.PARTICLES
for pname in NAME_TO_ID:
    ID_TO_NAME[NAME_TO_ID[pname]] = pname


def read_all_particles_from_first_event(path):
    with corsika_primary.particles.ParticleEventTapeReader(path) as pr:
        _, particle_block_reader = next(pr)
        particles = [block for block in particle_block_reader]
    return np.concatenate(particles)


def read_all_cherenkov_from_first_event(path):
    with corsika_primary.cherenkov.CherenkovEventTapeReader(path) as cr:
        _, cherenkov_block_reader = next(cr)
        cherenkov = [block for block in cherenkov_block_reader]
    return np.concatenate(cherenkov)


def particle_numeric_field_to_dict(particle):
    PAR = corsika_primary.I.PARTICLE
    out = {}
    out["corsika_particle_id"] = corsika_primary.particles.decode_particle_id(
        code=particle[PAR.CODE]
    )
    out["x_m"] = 1e-2 * particle[PAR.X]
    out["y_m"] = 1e-2 * particle[PAR.Y]
    out["momentum_xyz_GeV"] = [
        particle[PAR.PX],
        particle[PAR.PY],
        particle[PAR.PZ],
    ]

    return out


def split_particles_by_type(particles):
    out = {}
    for particle in particles:
        par = particle_numeric_field_to_dict(particle=particle)

        if par["corsika_particle_id"] in ID_TO_NAME:
            particle_name = ID_TO_NAME[par["corsika_particle_id"]]
            if particle_name in out:
                out[particle_name].append(par)
            else:
                out[particle_name] = [par]
    return out


def make_atmosphere_properties(altitude_asl_m):
    atm = {}
    atm["depth_g_per_cm2"] = tas.altitude_to_depth(altitude_asl_m)
    atm["refractive_index_1"] = tas.refraction_in_air(altitude_asl_m)

    beta_speed_of_light = 1.0
    atm["max_cherenkov_cone_half_angle_rad"] = np.arccos(
        1.0 / (atm["refractive_index_1"] * beta_speed_of_light)
    )

    dEdz_J_per_m = tas.dE_over_dz(
        q=tas.UNIT_CHARGE,
        beta=beta_speed_of_light,
        n=tas.refraction_in_air(altitude_asl_m),
        mu=tas.PERMABILITY_AIR,
        wavelength_start=250e-9,
        wavelength_end=700e-9,
    )
    dEdz_eV_per_m = dEdz_J_per_m / tas.UNIT_CHARGE

    atm["unit_charge_energy_to_cherenkov_light_dEdZ_eV_per_m"] = dEdz_eV_per_m

    return atm


def histogram_cherenkov_bunches(cherenkov_bunches):
    CER = corsika_primary.I.BUNCH
    cer_x_m = 1e-2 * cherenkov_bunches[:, CER.X_CM]
    cer_y_m = 1e-2 * cherenkov_bunches[:, CER.Y_CM]
    intensity = np.histogram2d(
        x=cer_x_m, y=cer_y_m, bins=(XY_BIN_EDGES_M, XY_BIN_EDGES_M)
    )[0]
    return intensity


def gather_statistics_of_cherenkov_bunches(cherenkov_bunches):
    CER = corsika_primary.I.BUNCH
    bunches = cherenkov_bunches
    sts = {}
    sts["size"] = np.sum(bunches[:, CER.BUNCH_SIZE_1])
    sts["average_wavelength_nm"] = np.average(
        np.abs(bunches[:, CER.WAVELENGTH_NM])
    )
    avg_wvl_m = 1e-9 * sts["average_wavelength_nm"]
    avg_ene_J = tas.PLANCK_ACTION * tas.SPEED_OF_LIGHT / avg_wvl_m
    sts["average_energy_eV"] = avg_ene_J / tas.UNIT_CHARGE
    return sts


steps = []
for iobs in range(len(observation_levels_asl_m)):
    iobs_key = "{:06d}".format(iobs)
    path = os.path.join(out_dir, iobs_key)
    particles = read_all_particles_from_first_event(path + ".particles.tar")
    cherenkov_bunches = read_all_cherenkov_from_first_event(
        path + ".cherenkov.tar"
    )

    step = {}
    step["step"] = iobs
    step["observation_level_asl_m"] = observation_levels_asl_m[iobs]
    step["time_to_detection_us"] = 1e6 * (
        (step["observation_level_asl_m"] - observation_level_hess_asl_m)
        / tas.SPEED_OF_LIGHT
    )
    step["particles"] = split_particles_by_type(particles=particles)
    step["cherenkov"] = {}
    step["cherenkov"]["statistics"] = gather_statistics_of_cherenkov_bunches(
        cherenkov_bunches=cherenkov_bunches
    )
    step["cherenkov"][
        "intensity_on_observation_level"
    ] = histogram_cherenkov_bunches(cherenkov_bunches=cherenkov_bunches)
    step["atmosphere"] = make_atmosphere_properties(
        altitude_asl_m=step["observation_level_asl_m"]
    )

    steps.append(step)

overview = {}
overview["steps"] = steps
overview["input"] = {}
overview["input"]["primary_energy_GeV"] = PRIMARY["energy_GeV"]
overview["input"]["primary_particle_type"] = ID_TO_NAME[PRIMARY["particle_id"]]
overview["input"]["observation_levels_asl_m"] = observation_levels_asl_m
overview["input"].update(namibia)
overview["input"]["random_seed"] = RANDOM_SEED
overview["input"]["xy_bin_edges_m"] = XY_BIN_EDGES_M


with open(os.path.join(out_dir, "overview.json"), "wt") as f:
    f.write(json_numpy.dumps(overview, indent=None))
