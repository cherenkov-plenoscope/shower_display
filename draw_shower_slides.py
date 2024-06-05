import svg_cartesian_plot as svgplt
import numpy
import json
import os
import sys

path = sys.argv[1]
out_dir = "airshower_display"

os.makedirs(out_dir, exist_ok=True)

with open(path, "rt") as f:
    shower = json.loads(f.read())

prng = np.random.Generator(np.random.PCG64(1))

scale = 1000 / 300

fontsize = "10px"
linespacing = 15

ABOUT_HALF_OF_CHERENKOV_PHOTONS_ARE_OUTSIDE_OF_MAIN_RING = 0.5

for step in shower["steps"]:
    fig = svgplt.Fig(cols=800, rows=800)
    ax = svgplt.Ax(fig)
    ax["span"] = (0, 0, 1, 1)
    ax["xlim"] = (-400, 400)
    ax["ylim"] = (-400, 400)

    ax_text = svgplt.Ax(fig)
    ax_text["span"] = (0.05, 0.05, 0.2, 0.2)
    ax_text["xlim"] = (0, 160)
    ax_text["ylim"] = (0, 160)

    ax_cerangle = svgplt.Ax(fig)
    ax_cerangle["span"] = (0.9, 0.1, 0.1, 0.8)
    ax_cerangle["xlim"] = (-40, 40)
    ax_cerangle["ylim"] = (0, 640)

    rr = step["in_plane_cherenkov_radius_m"] * scale
    ss = 0.2
    if rr > 0:
        circle_path = []
        for phi in np.linspace(0, 2 * np.pi, 101, endpoint=False):
            x = rr * np.cos(phi)
            y = rr * np.sin(phi)
            circle_path.append([x, y])

        svgplt.ax_add_path(
            ax=ax,
            xy=circle_path,
            fill=svgplt.color.css("blue"),
            fill_opacity=0.5
            * step["relative_in_plane_cherenkov_density_per_m2"],
        )

    num_gammas = step["num_gammas"]
    num_electrons = step["num_electrons"] // 2
    num_positrons = step["num_electrons"] - num_electrons

    print(num_gammas, num_electrons, num_positrons)

    for ne in range(num_electrons):
        xe = prng.normal(loc=0.0, scale=ss * rr)
        ye = prng.normal(loc=0.0, scale=ss * rr)
        svgplt.ax_add_text(
            ax=ax,
            text="e" + svgplt.text.superscript("-"),
            xy=[xe, ye],
            font_size=fontsize,
            font_family="Quicksand",
            fill=svgplt.color.css("green"),
        )

    for ne in range(num_positrons):
        xe = prng.normal(loc=0.0, scale=ss * rr)
        ye = prng.normal(loc=0.0, scale=ss * rr)
        svgplt.ax_add_text(
            ax=ax,
            text="e" + svgplt.text.superscript("+"),
            xy=[xe, ye],
            font_size=fontsize,
            font_family="Quicksand",
            fill=svgplt.color.css("red"),
        )

    for ne in range(num_gammas):
        xe = prng.normal(loc=0.0, scale=ss * rr)
        ye = prng.normal(loc=0.0, scale=ss * rr)
        svgplt.ax_add_text(
            ax=ax,
            text=svgplt.text.gamma(),
            xy=[xe, ye],
            font_size=fontsize,
            font_family="Quicksand",
            fill=svgplt.color.css("black"),
        )

    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "altitude above sea level: {: 6.3f}".format(
                1e-3 * step["altitude_m"]
            )
            + svgplt.text.thinspace()
            + "km"
        ),
        xy=[0, 8 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "time until detection: {: 6.1f}".format(
                step["time_to_detection_us"]
            )
            + svgplt.text.thinspace()
            + svgplt.text.mu()
            + "s"
        ),
        xy=[0, 7 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "atmospheric depth: {: 6.1f}".format(step["depth_g_per_cm2"])
            + svgplt.text.thinspace()
            + "g"
            + svgplt.text.thinspace()
            + "cm"
            + svgplt.text.superscript("-2")
        ),
        xy=[0, 6 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text="refractive index of air: {: 9.6f}".format(
            step["refraction_air_1"]
        ),
        xy=[0, 5 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text="number of gamma rays: {: 6d}".format(step["num_gammas"]),
        xy=[0, 4 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text="number of electrons and positrons: {: 6d}".format(
            step["num_electrons"]
        ),
        xy=[0, 3 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text="number of Cherenkov photons: {: 6d}".format(
            step["num_cherenkov"]
        ),
        xy=[0, 2 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "areal density of Cherenkov photons: "
            + "{: 9.1f}".format(
                step["in_plane_cherenkov_density_per_m2"]
                * ABOUT_HALF_OF_CHERENKOV_PHOTONS_ARE_OUTSIDE_OF_MAIN_RING
            )
            + svgplt.text.thinspace()
            + "m"
            + svgplt.text.superscript("-2")
        ),
        xy=[0, 1 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "energy loss of electron or positron to Cherenkov light: "
            + "{: 6.1f}".format(
                step["dE_over_dz_electron_speed_c_in_eV_per_m"]
            )
            + svgplt.text.thinspace()
            + "eV"
            + svgplt.text.thinspace()
            + "m"
            + svgplt.text.superscript("-1")
        ),
        xy=[0, 0 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )

    # Cherenkov angle display
    # -----------------------
    cer_half_angle_rad = step["max_cherenkov_angle_rad"]
    cer_full_angle_deg = 2.0 * np.rad2deg(cer_half_angle_rad)
    cer_length = 640
    cer_start = [0, cer_length]
    cer_stop_1 = [-np.tan(cer_half_angle_rad) * cer_length, 0]
    cer_stop_2 = [+np.tan(cer_half_angle_rad) * cer_length, 0]
    svgplt.ax_add_path(
        ax=ax_cerangle,
        xy=[cer_start, cer_stop_1],
        stroke=svgplt.color.css("black"),
        stroke_opacity=0.5,
    )
    svgplt.ax_add_path(
        ax=ax_cerangle,
        xy=[cer_start, cer_stop_2],
        stroke=svgplt.color.css("black"),
        stroke_opacity=0.5,
    )
    svgplt.ax_add_text(
        ax=ax_cerangle,
        text=(
            "Cherenkov cone full angle: "
            + "{: 6.2f}".format(cer_full_angle_deg)
            + svgplt.text.thinspace()
            + svgplt.text.circ()
        ),
        xy=[-140, -1 * linespacing],
        font_size=fontsize,
        font_family="Quicksand",
    )

    # H.E.S.S. array
    # --------------
    hess_array_angle = np.deg2rad(45)
    hess_square_radius = np.sqrt(1 / 2) * 120
    hess_ct1to4_radius = 6
    hess_ct5_radius = 14

    if step["step"] == 0:
        svgplt.shapes.ax_add_circle(
            ax=ax,
            xy=[0, 0],
            radius=scale * hess_ct5_radius,
            fill=svgplt.color.css("red"),
        )
        svgplt.ax_add_text(
            ax=ax,
            text=("H.E.S.S. CT-5"),
            xy=[-0.5 * scale * hess_ct5_radius, -0.5 * linespacing],
            font_size=fontsize,
            font_family="Quicksand",
        )

        for ij, phi in enumerate(np.linspace(0, 2 * np.pi, 4, endpoint=False)):
            hess_ct_x = (
                scale * hess_square_radius * np.cos(phi + hess_array_angle)
            )
            hess_ct_y = (
                scale * hess_square_radius * np.sin(phi + hess_array_angle)
            )
            svgplt.shapes.ax_add_circle(
                ax=ax,
                xy=[hess_ct_x, hess_ct_y],
                radius=scale * hess_ct1to4_radius,
                fill=svgplt.color.css("red"),
            )
            svgplt.ax_add_text(
                ax=ax,
                text=("H.E.S.S. CT-{:d}".format(1 + ij)),
                xy=[
                    hess_ct_x - 0.5 * scale * hess_ct1to4_radius,
                    hess_ct_y - 0.5 * linespacing,
                ],
                font_size=fontsize,
                font_family="Quicksand",
            )

    svgplt.fig_write(
        fig, os.path.join(out_dir, "{:03d}.svg".format(step["step"]))
    )


index_html = ""
index_html += "<!DOCTYPE html>\n"

index_html += '<html lang="en">\n'
index_html += "  <head>\n"
index_html += '    <meta charset="utf-8">\n'
index_html += "    <title>Air shower display</title>\n"
index_html += '    <link rel="stylesheet" href="style.css">\n'
index_html += "  </head>\n"
index_html += "  <body>\n"
index_html += "  <h1>Air shower display</h1>\n"
for back in range(len(shower["steps"])):
    istep = len(shower["steps"]) - back - 1
    step = shower["steps"][istep]
    index_html += "    <h2>Step: {: 2d}, position in display: {:3.1f}m, in atmosphere {:3.1f}km</h2>\n".format(
        step["step"],
        (step["altitude_m"] - 1.8e3) * scale * 1e-3,
        step["altitude_m"] * 1e-3,
    )
    index_html += '      <img src="{:s}" alt="step" width="800px" height="800px">\n'.format(
        "{:03d}.svg".format(step["step"])
    )
index_html += "  </body>\n"
index_html += "</html>\n"

with open(os.path.join(out_dir, "index.html"), "wt") as f:
    f.write(index_html)
