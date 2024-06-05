import toy_air_shower as tas
import json_numpy as json
import os
import pickle

primary_energy_GeV = 100
wavelength_start = 250e-9  # m
wavelength_end = 700e-9  # m
primary_energy = primary_energy_GeV * 1e9 * tas.UNIT_CHARGE  # J

random_seed = 1
result_dir = "{:03d}.shower".format(random_seed)


def write_shower(path, particles, cherenkov_photons):
    os.makedirs(path)
    with open(os.path.join(path, "particles.json"), "wt") as f:
        f.write(json.dumps(particles))
    with open(os.path.join(path, "cherenkov-photons.pickle"), "wb") as f:
        f.write(pickle.dumps(cherenkov_photons))


def read_shower(path):
    with open(os.path.join(path, "particles.json"), "rt") as f:
        particles = json.loads(f.read())
    with open(os.path.join(path, "cherenkov-photons.pickle"), "rb") as f:
        cherenkov_photons = pickle.loads(f.read())
    return particles, cherenkov_photons


if not os.path.exists(result_dir):
    particles, cherenkov_photons = tas.simulate_gamma_ray_air_shower(
        random_seed=random_seed,
        primary_energy=primary_energy,
        wavelength_start=wavelength_start,
        wavelength_end=wavelength_end,
    )
    write_shower(
        path=result_dir,
        particles=particles,
        cherenkov_photons=cherenkov_photons,
    )

particles, cherenkov_photons = read_shower(path=result_dir)


scale = 1 / 300


cherenkov_radius_on_ground_m = 110

altitude_start_m = 1.8e3
altitude_range_m = 30e3
altitude_steps_m = np.geomspace(
    altitude_start_m, altitude_start_m + altitude_range_m, 8
)

steps = []

for a in range(len(altitude_steps_m)):
    altitude_m = altitude_steps_m[a]

    ratio = 1 - (altitude_m - altitude_start_m) / altitude_range_m

    in_plane_cherenkov_radius_m = cherenkov_radius_on_ground_m * ratio

    num_electrons = 0
    num_gammas = 0
    num_cherenkov = 0
    E_gamma_J = 0
    E_electron_J = 0

    for p in particles:
        if p["start_altitude"] > altitude_m and p["end_altitude"] < altitude_m:
            if p["type"] == "gamma":
                num_gammas += 1
                E_gamma_J += p["start_energy"]
            elif p["type"] == "electron":
                num_electrons += 1
                E_electron_J += p["start_energy"]

    if num_gammas > 0:
        E_gamma_J = E_gamma_J / num_gammas
    else:
        E_gamma_J = float("nan")

    if num_electrons > 0:
        E_electron_J = E_electron_J / num_electrons
    else:
        E_electron_J = float("nan")

    cherenkov_photon_emission_altitude = cherenkov_photons[:, tas.IDX_ALTITUDE]
    cherenkov_wavelength = cherenkov_photons[:, tas.IDX_WAVELENGTH]

    cherenkov_mask = cherenkov_photon_emission_altitude > altitude_m

    num_cherenkov = np.sum(cherenkov_mask)

    avg_cherenkov_wavelength = np.average(cherenkov_wavelength[cherenkov_mask])
    avg_cherenkov_energy_J = (
        tas.PLANCK_ACTION * tas.SPEED_OF_LIGHT / avg_cherenkov_wavelength
    )
    avg_cherenkov_energy_eV = avg_cherenkov_energy_J / tas.UNIT_CHARGE

    beta_speed_of_light = 1.0

    dE_over_dz_electron_in_J_per_m = tas.dE_over_dz(
        q=tas.UNIT_CHARGE,
        beta=beta_speed_of_light,
        n=tas.refraction_in_air(altitude_m),
        mu=tas.PERMABILITY_AIR,
        wavelength_start=wavelength_start,
        wavelength_end=wavelength_end,
    )
    dE_over_dz_electron_in_eV_per_m = (
        dE_over_dz_electron_in_J_per_m / tas.UNIT_CHARGE
    )

    max_cherenkov_angle_rad = np.arccos(
        1.0 / (tas.refraction_in_air(altitude_m) * beta_speed_of_light)
    )

    step = {
        "step": a,
        "altitude_m": altitude_m,
        "num_gammas": num_gammas,
        "num_electrons": num_electrons,
        "avg_gamma_energy_GeV": 1e-9 * E_gamma_J / tas.UNIT_CHARGE,
        "avg_electron_energy_GeV": 1e-9 * E_electron_J / tas.UNIT_CHARGE,
        "avg_cherenkov_wavelength_nm": 1e9 * avg_cherenkov_wavelength,
        "avg_cherenkov_energy_eV": avg_cherenkov_energy_eV,
        "total_cherenkov_energy_MeV": 1e-6
        * avg_cherenkov_energy_eV
        * num_cherenkov,
        "num_cherenkov": num_cherenkov,
        "depth_g_per_cm2": tas.altitude_to_depth(altitude_m),
        "refraction_air_1": tas.refraction_in_air(altitude_m),
        "time_to_detection_us": (
            (altitude_m - altitude_start_m) / tas.SPEED_OF_LIGHT
        )
        * 1e6,
        "in_plane_cherenkov_radius_m": in_plane_cherenkov_radius_m,
        "in_plane_cherenkov_density_per_m2": num_cherenkov
        / (np.pi * in_plane_cherenkov_radius_m**2),
        "dE_over_dz_electron_speed_c_in_eV_per_m": dE_over_dz_electron_in_eV_per_m,
        "max_cherenkov_angle_rad": max_cherenkov_angle_rad,
    }
    steps.append(step)


max_in_plane_cherenkov_density_per_m2 = max(
    [s["in_plane_cherenkov_density_per_m2"] for s in steps]
)

for s in steps:
    s["relative_in_plane_cherenkov_density_per_m2"] = (
        s["in_plane_cherenkov_density_per_m2"]
        / max_in_plane_cherenkov_density_per_m2
    )

overview = {
    "primary_energy_GeV": primary_energy_GeV,
    "primary_particle_type": "gamma",
    "model": "J. Matthews, 'A Heitler model of extensive air showers' Astroparticle Physics 22 (2005) 387-397",
    "steps_altitudes_m": altitude_steps_m,
    "steps": steps,
}

with open(os.path.join(result_dir, "overview.json"), "wt") as f:
    f.write(json.dumps(overview, indent=4))
