###############################################
Create displays of atmospheric particle showers
###############################################

This is a collection of loose python scripts which are used to manufacture
displays of atmospheric particle showers for public events.

The scripts require python packages from
https://github.com/cherenkov-plenoscope which must be installed first.

- corsika_primary (needs KIT CORSIKA credentials or ``corsika-75600.tar.gz``)
- toy_air_shower
- json_numpy
- svg_cartesian_plot

*******
concept
*******

|shower_slide|

Example slide of a shower induced by a 250GeV gamma ray over the sky of Namibia.

The motivation here is to make a model which is easy to manufacture using e.g.
paper printing or laser engraving.
The airshower is shown using multiple 2D slices, perpendicular to the
longitudinal axis of the shower.
The slices show the denisty of the Cherenkov light and the particles.
To ease the fabrication with a laser engraver, the Cherenkov light's density is
shown using contour lines. Contour lines are bad for many reasons, but they
are easy to engrave with a laser.

.. |shower_slide| image:: https://github.com/cherenkov-plenoscope/shower_display/blob/main/readme/004.svg
