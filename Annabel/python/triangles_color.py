import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pandas as pd
from matplotlib.patches import Rectangle
from TVA1_UU import TVA1_UU

filename = "dvcs_xs_May-2021_342_sets_with_trueCFFs.csv"
xlabel = "x_b"
ylabel = "QQ"
zlabel = "t"
# set = 1
# xvar = np.arange(0.35, 0.37, 0.002)         #x_b set 1
# yvar = np.arange(1.4, 1.6, 0.02)            # QQ set 1
# zvar = np.arange(-0.31, -0.29, 0.002)       # t set 1
name_color_map = 'gist_rainbow'

def function(x, y, z):
    # x, y and z are the varying inputs
    inputs[xlabel] = np.full([n, 1], x)
    inputs[ylabel] = np.full([n, 1], y)
    inputs[zlabel] = np.full([n, 1], z)

    kinematics = np.concatenate((phi_x, inputs["k"], inputs["QQ"], inputs["x_b"], inputs["t"], inputs["F1"], inputs["F2"], inputs["dvcs"]), axis=1)
    F = tv.TotalUUXS(kinematics, inputs["ReH"], inputs["ReE"], inputs["ReHTilde"])
    area = F.numpy()[0, 0]
    return abs((area - f_data)/f_data)

full_df = pd.read_csv(filename)
tv = TVA1_UU()
# ax = plt.axes(projection='3d')

# xvar = np.arange(0.59, 0.61, 0.002)  #x_b set 5
# xvar = np.arange(0.44, 0.46, 0.0015)  #x_b set 10

# yvar = np.arange(1.5, 1.7, 0.02)    # QQ set 5
# yvar = np.arange(1.55, 1.75, 0.015)    # QQ set 10

# zvar = np.arange(-0.50, -0.48, 0.002)       # t set 5
# zvar = np.arange(-0.43, -0.41, 0.0015)       # t set 10

# ax.set_xlabel(xlabel)
# ax.set_ylabel(ylabel)
# ax.set_zlabel(zlabel)

for set in range(1, 343):
    df = full_df[full_df["#Set"] == set]

    phi_x = np.array(df[["phi_x"]])
    n = len(phi_x)
    inputs = dict()
    # inputs["ReH"] = np.full([n, 1], 1.60016) # 2.16161) # 0.60357
    # inputs["ReE"] = np.full([n, 1], -3.32832) # -2.46753) # 42.07553
    # inputs["ReHtilde"] = np.full([n, 1], 3.50066) # 4.18988) # 3.36188
    for label in ["k", "QQ", "x_b", "t", "F1", "F2", "dvcs", "ReH", "ReE", "ReHTilde"]:
        if label in df:
            inputs[label] = np.array(df[[label]])

    f_data = np.array(df[["F"]])[0]
    cx = inputs["x_b"][0]
    cy = inputs["QQ"][0]
    cz = inputs["t"][0]
    xvar = np.arange(cx - 0.01, cx + 0.01, 0.002)  # x_b
    yvar = np.arange(cy - 0.1, cy + 0.1, 0.02)  # QQ
    zvar = np.arange(cz - 0.01, cz + 0.01, 0.002)  # t

    X, Y, Z = np.meshgrid(xvar, yvar, zvar)
    f = np.vectorize(function)
    c = f(X, Y, Z)

    x = X.flatten()
    y = Y.flatten()
    z = Z.flatten()
    c = c.flatten()
    #end
    #-----

    ax = plt.axes(projection='3d')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    img = ax.scatter(X, Y, Z, s=10, c=c, cmap=name_color_map)
    cbar = plt.colorbar(img, pad=0.2)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label('|(F(calc) - F(data))/F(data)|')
    consts = []
    for key in inputs:
        if key == xlabel or key == ylabel or key == zlabel or key == "F1" or key == "F2":
            continue
        consts.append(key + " = " + str(inputs[key][0, 0]))
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    ax.legend([extra]*6, ("constants taken from set " + str(set), consts[0], consts[1], consts[2], consts[3], consts[4]), bbox_to_anchor=(0.2, 1.13), fontsize='xx-small')
    plt.savefig("set " + str(set) + "s.png")

    # We create triangles that join 3 pt at a time and where their colors will be
    # determined by the values ​​of their 4th dimension. Each triangle contains 3
    # indexes corresponding to the line number of the points to be grouped.
    # Therefore, different methods can be used to define the value that
    # will represent the 3 grouped points
    # triangles = mtri.Triangulation(x, y).triangles
    # colors = np.mean( [c[triangles[:,0]], c[triangles[:,1]], c[triangles[:,2]]], axis = 0)
    #
    # #end
    # #----------
    # # Displays the 4D graphic.
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # triang = mtri.Triangulation(x, y, triangles)
    # surf = ax.plot_trisurf(triang, z, cmap = name_color_map, shade=False, linewidth=0.2)
    # surf.set_array(colors)
    # surf.autoscale()
    #
    # #Add a color bar with a title to explain which variable is represented by the color.
    # cbar = fig.colorbar(surf, pad=0.2)
    # cbar.ax.get_yaxis().labelpad = 15
    # cbar.set_label('|(F(calc) - F(data))/F(data)|')
    #
    # # Add titles to the axes and a title in the figure.
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # ax.set_zlabel(zlabel)
    # # plt.title('|F(calc) - F(data)| in function of x_b, QQ and t')
    #
    # # Legend
    # consts = []
    # for key in inputs:
    #     if key == xlabel or key == ylabel or key == zlabel or key == "F1" or key == "F2":
    #         continue
    #     consts.append(key + " = " + str(inputs[key][0, 0]))
    # extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    # ax.legend([extra]*6, ("constants taken from set " + str(set), consts[0], consts[1], consts[2], consts[3], consts[4]), bbox_to_anchor=(0.2, 1.13), fontsize='xx-small')
    #
    # plt.savefig("set " + str(set) + "t.png")