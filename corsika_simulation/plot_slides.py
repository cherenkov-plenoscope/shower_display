import svg_cartesian_plot as svgplt
import numpy as np
import json_numpy
import os
import argparse
from skimage import measure as skimage_measure
from skimage import filters as skimage_filters

parser = argparse.ArgumentParser(
    prog="plot_slides",
    description="Plots slides of simulated shower.",
)
parser.add_argument(
    "shower_path",
    metavar="SHOWER_PATH",
    type=str,
    help=("The input shower path (json file)."),
)
parser.add_argument(
    "out_dir",
    metavar="OUT_DIR",
    type=str,
    help=("The of output directory."),
)
args = parser.parse_args()

out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)

with open(args.shower_path, "rt") as f:
    shower = json_numpy.loads(f.read())


SLIDE_RADIUS_MM = 400

scale = 1000 / 400

fontsize = "10px"
linespacing = 15
font_family = "Georgia, serif"
particle_font_family = "Georgia, serif"
particle_opacity = 0.5

cer_vmin = 0.0
cer_vmax = np.max(
    [
        step["cherenkov"]["intensity_on_observation_level"].max()
        for step in shower["steps"]
    ]
)

cer_colormap = svgplt.color.Map(
    name="Blues",
    start=cer_vmin,
    stop=cer_vmax,
    func=svgplt.scaling.power(0.333),
)

cer_film_intensity_num_bins = 32
cer_film_intensity_bin_edges = np.geomspace(
    cer_vmax * 0.005,
    cer_vmax,
    cer_film_intensity_num_bins + 1,
)

cherenkov_mode = "contours"
cherenkov_smoothing_sigma = 1.0


def font_size_px_from_energy(
    energy_GeV, font_size_px_at_1MeV=1, font_size_px_at_1000GeV=30
):
    log10 = np.log10
    x = log10(energy_GeV)
    xp = [log10(0.001), log10(1000)]
    fp = [font_size_px_at_1MeV, font_size_px_at_1000GeV]
    return np.interp(x=x, xp=xp, fp=fp)


def particle_avg_energy(particles):
    return np.average(
        [np.linalg.norm(e["momentum_xyz_GeV"]) for e in particles]
    )


def contour_is_closed(contour):
    return contour[0][0] == contour[-1][0] and contour[0][1] == contour[-1][1]


def find_contours(image, overhead=5, **kwargs):
    assert overhead > 0
    mat = np.zeros(
        shape=(
            image.shape[0] + 2 * overhead,
            image.shape[1] + 2 * overhead,
        )
    )
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            mat[x + overhead, y + overhead] = image[x, y]

    contours_px = skimage_measure.find_contours(image=mat, **kwargs)
    out = []
    for contour_px in contours_px:
        o = []
        for xy_px in contour_px:
            x_px = xy_px[0]
            y_px = xy_px[1]
            o.append([x_px - overhead, y_px - overhead])
        out.append(np.array(o))
    return out


def energy_str(energy_eV, float_format="{:.1f}"):
    if energy_eV > 1e12:
        prefix = "T"
        val = 1e-12 * energy_eV
    elif energy_eV > 1e9:
        prefix = "G"
        val = 1e-9 * energy_eV
    elif energy_eV > 1e6:
        prefix = "M"
        val = 1e-6 * energy_eV
    else:
        prefix = ""
        val = energy_eV
    return float_format.format(val) + svgplt.text.thinspace() + prefix + "eV"


def contour_area(xy):
    """
    Shoelace formula
    """
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def make_cer_film_layers(
    intensity_on_observation_level,
    intensity_bin_edges,
    xy_bin_edges,
    area_threshold=10,
):
    layers = []
    _num_pix = len(xy_bin_edges) - 1
    _unit_per_px = (max(xy_bin_edges) - min(xy_bin_edges)) / _num_pix

    for intensity_bin in range(len(intensity_bin_edges) - 1):
        istart = intensity_bin_edges[intensity_bin]
        contours_px = find_contours(
            image=intensity_on_observation_level,
            level=istart,
            fully_connected="high",
            positive_orientation="high",
        )
        layer = []
        for c in range(len(contours_px)):
            contour_px = contours_px[c]
            contour = []
            for i in range(len(contour_px)):
                x_px = contour_px[i][0]
                y_px = contour_px[i][1]
                x = x_px * _unit_per_px + min(xy_bin_edges)
                y = y_px * _unit_per_px + min(xy_bin_edges)
                contour.append([x, y])
            contour = np.array(contour)
            if contour_is_closed(contour):
                area = contour_area(xy=contour)
                if area > area_threshold:
                    layer.append(contour)

        layers.append(layer)
    return layers


def add_to_ax_particle(ax, scale, particle, **kwargs):
    margin_mm = 10
    xe = scale * particle["x_m"]
    ye = scale * particle["y_m"]
    energy_GeV = np.linalg.norm(particle["momentum_xyz_GeV"])
    if xe > ax["xlim"][0] + margin_mm and xe < ax["xlim"][1] - margin_mm:
        if ye > ax["ylim"][0] + margin_mm and ye < ax["ylim"][1] - margin_mm:
            font_size_px = font_size_px_from_energy(energy_GeV=energy_GeV)
            if True:
                ff = 0.0
            else:
                ff = 0.2
                svgplt.shapes.ax_add_circle(
                    ax=ax,
                    xy=[xe, ye],
                    radius=font_size_px,
                    stroke=svgplt.color.css("black"),
                    fill=None,
                )
            svgplt.ax_add_text(
                ax=ax,
                xy=[xe - ff * font_size_px, ye - ff * font_size_px],
                font_size="{:f}px".format(font_size_px),
                **kwargs,
            )


def add_plot_hess_array(ax, scale):
    hess_array_angle = np.deg2rad(45)
    hess_square_radius = np.sqrt(1 / 2) * 120
    hess_ct1to4_radius = 6
    hess_ct5_radius = 14

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
        font_family=font_family,
    )

    for ij, phi in enumerate(np.linspace(0, 2 * np.pi, 4, endpoint=False)):
        hess_ct_x = scale * hess_square_radius * np.cos(phi + hess_array_angle)
        hess_ct_y = scale * hess_square_radius * np.sin(phi + hess_array_angle)
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
            font_family=font_family,
        )


def flippy(mat):
    out = np.zeros(shape=mat.shape)
    xl = mat.shape[0]
    yl = mat.shape[1]
    for x in range(xl):
        for y in range(yl):
            out[yl - y - 1, xl - x - 1] = mat[x, y]
    return out


def ax_add_symetry_grid(ax):
    svgplt.ax_add_path(
        ax=ax,
        xy=[[ax["xlim"][0], ax["ylim"][0]], [ax["xlim"][1], ax["ylim"][1]]],
        stroke=svgplt.color.css("black"),
    )
    svgplt.ax_add_path(
        ax=ax,
        xy=[[ax["xlim"][0], ax["ylim"][1]], [ax["xlim"][1], ax["ylim"][0]]],
        stroke=svgplt.color.css("black"),
    )
    svgplt.ax_add_path(
        ax=ax,
        xy=[
            [(ax["xlim"][0] + ax["xlim"][1]) / 2, ax["ylim"][0]],
            [(ax["xlim"][0] + ax["xlim"][1]) / 2, ax["ylim"][1]],
        ],
        stroke=svgplt.color.css("black"),
    )
    svgplt.ax_add_path(
        ax=ax,
        xy=[
            [ax["xlim"][0], (ax["ylim"][0] + ax["ylim"][1]) / 2],
            [ax["xlim"][1], (ax["ylim"][0] + ax["ylim"][1]) / 2],
        ],
        stroke=svgplt.color.css("black"),
    )


for step in shower["steps"]:
    print("step", step["step"])
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

    cherenkov_bitmap = skimage_filters.gaussian(
        image=step["cherenkov"]["intensity_on_observation_level"],
        sigma=cherenkov_smoothing_sigma,
    )

    if cherenkov_mode == "bitmap":
        svgplt.ax_add_pcolormesh(
            ax=ax,
            z=flippy(cherenkov_bitmap),
            colormap=cer_colormap,
            x_bin_edges=shower["input"]["xy_bin_edges_m"] * scale,
            y_bin_edges=shower["input"]["xy_bin_edges_m"] * scale,
        )

    elif cherenkov_mode == "contours":
        cer_layers = make_cer_film_layers(
            intensity_on_observation_level=cherenkov_bitmap,
            intensity_bin_edges=cer_film_intensity_bin_edges,
            xy_bin_edges=shower["input"]["xy_bin_edges_m"] * scale,
            area_threshold=1000,
        )

        for ii in range(0, cer_film_intensity_num_bins):
            print("layer", ii)
            for contour_xy in cer_layers[ii]:
                if len(contour_xy) > 0:
                    svgplt.ax_add_path(
                        ax=ax,
                        xy=contour_xy,
                        fill=None,
                        # fill=svgplt.color.css("blue"),
                        # fill_opacity=1/cer_film_intensity_num_bins,
                        stroke=svgplt.color.css("black"),
                        stroke_width=0.25,
                    )

    # ax_add_symetry_grid(ax=ax)

    if step["step"] == 0:
        add_plot_hess_array(ax=ax, scale=scale)

    if "electron" in step["particles"]:
        for particle in step["particles"]["electron"]:
            add_to_ax_particle(
                ax=ax,
                scale=scale,
                particle=particle,
                text="e" + svgplt.text.superscript("-"),
                font_family=particle_font_family,
                opacity=particle_opacity,
                fill=svgplt.color.css("black"),
            )

    if "positron" in step["particles"]:
        for particle in step["particles"]["positron"]:
            add_to_ax_particle(
                ax=ax,
                scale=scale,
                particle=particle,
                text="e" + svgplt.text.superscript("+"),
                font_family=particle_font_family,
                opacity=particle_opacity,
                fill=svgplt.color.css("black"),
            )

    if "gamma" in step["particles"]:
        for particle in step["particles"]["gamma"]:
            add_to_ax_particle(
                ax=ax,
                scale=scale,
                particle=particle,
                text=svgplt.text.gamma(),
                font_family=particle_font_family,
                opacity=particle_opacity,
                fill=svgplt.color.css("black"),
            )

    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "altitude above sea level: {: 6.3f}".format(
                1e-3 * step["observation_level_asl_m"]
            )
            + svgplt.text.thinspace()
            + "km"
        ),
        xy=[0, 8 * linespacing],
        font_size=fontsize,
        font_family=font_family,
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
        font_family=font_family,
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "atmospheric depth: {: 6.1f}".format(
                step["atmosphere"]["depth_g_per_cm2"]
            )
            + svgplt.text.thinspace()
            + "g"
            + svgplt.text.thinspace()
            + "cm"
            + svgplt.text.superscript("-2")
        ),
        xy=[0, 6 * linespacing],
        font_size=fontsize,
        font_family=font_family,
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text="refractive index of air: {: 9.6f}".format(
            step["atmosphere"]["refractive_index_1"]
        ),
        xy=[0, 5 * linespacing],
        font_size=fontsize,
        font_family=font_family,
    )

    if "gamma" in step["particles"]:
        _num = len(step["particles"]["gamma"])
        _avg_ene_GeV = particle_avg_energy(step["particles"]["gamma"])
        svgplt.ax_add_text(
            ax=ax_text,
            text=(
                svgplt.text.gamma()
                + " number: {: 6d},".format(_num)
                + " avg. energy: "
                + energy_str(energy_eV=1e9 * _avg_ene_GeV)
            ),
            xy=[0, 4 * linespacing],
            font_size=fontsize,
            font_family=font_family,
        )
    if "electron" in step["particles"]:
        _num = len(step["particles"]["electron"])
        _avg_ene_GeV = particle_avg_energy(step["particles"]["electron"])
        svgplt.ax_add_text(
            ax=ax_text,
            text=(
                "e"
                + svgplt.text.superscript("-")
                + " number: {: 6d},".format(_num)
                + " avg. energy: "
                + energy_str(energy_eV=1e9 * _avg_ene_GeV)
            ),
            xy=[0, 3 * linespacing],
            font_size=fontsize,
            font_family=font_family,
        )
    if "positron" in step["particles"]:
        _num = len(step["particles"]["positron"])
        _avg_ene_GeV = particle_avg_energy(step["particles"]["positron"])
        svgplt.ax_add_text(
            ax=ax_text,
            text=(
                "e"
                + svgplt.text.superscript("+")
                + " number: {: 6d},".format(_num)
                + " avg. energy: "
                + energy_str(energy_eV=1e9 * _avg_ene_GeV)
            ),
            xy=[0, 2 * linespacing],
            font_size=fontsize,
            font_family=font_family,
        )

    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "Cherenkov photons"
            + " number: {: 6d},".format(
                int(step["cherenkov"]["statistics"]["size"])
            )
            + " avg. energy: "
            + energy_str(
                energy_eV=step["cherenkov"]["statistics"]["average_energy_eV"]
            )
            + ", avg. wavelength: {:.1f}".format(
                step["cherenkov"]["statistics"]["average_wavelength_nm"]
            )
            + svgplt.text.thinspace()
            + "nm,"
        ),
        xy=[0, 1 * linespacing],
        font_size=fontsize,
        font_family=font_family,
    )
    svgplt.ax_add_text(
        ax=ax_text,
        text=(
            "energy loss of electron or positron to Cherenkov light: "
            + "{: 6.1f}".format(
                step["atmosphere"][
                    "unit_charge_energy_to_cherenkov_light_dEdZ_eV_per_m"
                ]
            )
            + svgplt.text.thinspace()
            + "eV"
            + svgplt.text.thinspace()
            + "m"
            + svgplt.text.superscript("-1")
        ),
        xy=[0, 0 * linespacing],
        font_size=fontsize,
        font_family=font_family,
    )

    # Cherenkov angle display
    # -----------------------
    cer_half_angle_rad = step["atmosphere"][
        "max_cherenkov_cone_half_angle_rad"
    ]
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
        font_family=font_family,
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
        (step["observation_level_asl_m"] - 1.8e3) * scale * 1e-3,
        step["observation_level_asl_m"] * 1e-3,
    )
    index_html += '      <img src="{:s}" alt="step" width="800px" height="800px">\n'.format(
        "{:03d}.svg".format(step["step"])
    )
index_html += "  </body>\n"
index_html += "</html>\n"

with open(os.path.join(out_dir, "index.html"), "wt") as f:
    f.write(index_html)
