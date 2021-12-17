# Graphs in Mathematica

**official_va_formulation** and 
**official_bkm_formulation**

**part1_kinematics_vs_cross_section_graphs**, 
**part2_CFFs_vs_cross_section_graphs**, and 
**part3_derivative_graphs** create 3d plots using the VA formulation with different combinations of axes

# Graphs in Python matplotlib

 * constant values of kinematics, CFF's, etc. are from a given #Set in datafile
 * range of values for independent variable (x-axis) depend on the range in datafile
 * (or hard coded for CFFs)

**2d_plotting** 
creates a 2d scatterplot with [k, QQ, x_b, t, ReH, ReE, ReHtilde, or phi]
on the x-axis and F on the y-axis

**2d_slider_model**
creates a 2d scatterplot with total cross-section (F) on the y-axis and phi on the x-axis, and sliders controlling the values of [k, QQ, x, t, F1, F2, dvcs, ReH, ReE, and ReHtilde]

**3d_plotting**
creates a 3D scatterplot with [k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs]
on the x- and y-axis and the integral of F w/ respect to phi on the z-axis

**3d_plotting_color**
creates a 3d scatterplot allowing for input from [k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs] for x-, y-, and z-axis
while color varies according to the total cross-section

**triangles_color**
creates a 3d plot of Delauney triangulation with color varying according to the total cross-section and any three axes from the variables above
