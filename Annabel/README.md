# Graphing in Mathematica

official_va_formulation
official_bkm_formulation
va_formulation_test
bkm_formulation_test

part1_kinematics_vs_cross_section_graphs
part2_CFFs_vs_cross_section_graphs
part3_derivative_graphs

# 2D and 3D graphing in Python matplotlib

**2d_plotting** 
creates a 2d scatterplot with [k, QQ, x_b, t, ReH, ReE, ReHtilde, or phi]
on the x-axis and F on the y-axis

 * constant values of kinematics, CFF's, etc. are from a given #Set in datafile
 * range of values for x-axis depend on the range in datafile
 * (or hard coded for CFFs)

**2d_slider_model**
creates a 2d scatterplot with sliders controlling the values of k, QQ, x, t, F1, F2, dvcs, ReH, ReE, and ReHtilde
 * initial values and ranges hardcoded based on ranges in the dataset "dvcs_xs_newsets_genCFFs"
 * 2d scatterplot has total cross-section (F) on the y-axis and phi on the x-axis

**3d_plotting**
creates a 3D scatterplot with [k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs]
on the x- and y-axis and the integral of F w/ respect to phi on the z-axis

 * constant values of kinematics, CFF's, etc. are from any #Set in datafile
 * range of values for x-axis depend on the range in datafile
 * (or, if unavailable, hard coded for CFF's)

**3d_plotting_color**
**triangles_color**
