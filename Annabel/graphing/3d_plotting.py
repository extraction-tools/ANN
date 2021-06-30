import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TVA1_UU import TVA1_UU
from BHDVCStf import BHDVCS

"""
creates a 3D scatterplot with [k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs]
on the x- and y-axis and the integral of F w/ respect to phi on the z-axis

 * constant values of kinematics, CFF's, etc. are from any #Set in datafile
 * range of values for x-axis depend on the range in datafile
 * (or, if unavailable, hard coded for CFF's)

FIELDS -
line 51: #Set in datafile
line 52: datafile name
line 53: formulation, TVA1_UU vs. BHDVCS
line 60: step size of x-axis
line 74: step size of y-axis
"""

def function(x, y):
    # x and y are the varying inputs
    phi_x = np.array(df[["phi_x"]])
    n = len(phi_x)
    inputs = dict()
    inputs["ReH"] = np.full([n, 1], 9.6)
    inputs["ReE"] = np.full([n, 1], -49.6)
    inputs["ReHtilde"] = np.full([n, 1], 5.3)
    for label in ["k", "QQ", "x_b", "t", "F1", "F2", "dvcs", "ReH", "ReE", "ReHtilde"]:
        if label in df:
            inputs[label] = np.array(df[[label]])

    inputs[xlabel] = np.full([n, 1], x)
    inputs[ylabel] = np.full([n, 1], y)

    kinematics = np.concatenate((phi_x, inputs["k"], inputs["QQ"], inputs["x_b"], inputs["t"], inputs["F1"], inputs["F2"], inputs["dvcs"]), axis=1)
    F = tv.TotalUUXS(kinematics, inputs["ReH"], inputs["ReE"], inputs["ReHtilde"])
    area = 0.0
    phi_step = phi_x[1, 0] - phi_x[0, 0]
    for val in F.numpy()[0, :]:
        area = area + (val * phi_step)
    return area

xlabel = input("Enter variable on x-axis (k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs): ")
ylabel = input("Enter variable on y-axis (k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs): ")
xlabel.strip()
ylabel.strip()

set = 1
df = pd.read_csv("dvcs_May21.csv")
tv = TVA1_UU()
ax = plt.axes(projection='3d')

# range of x axis
if xlabel in df:
    min = np.amin(df[xlabel])
    max = np.amax(df[xlabel])
    step = (max - min)/25
    xvar = np.arange(min, max+step, step)
else:
    if xlabel == "ReH":
        xvar = np.arange(-5, 31, 1)
    elif xlabel == "ReE":
        xvar = np.arange(-71, -35, 1)
    else:
        xvar = np.arange(-11, 25, 1)

# range of y axis
if ylabel in df:
    min = np.amin(df[xlabel])
    max = np.amax(df[xlabel])
    step = (max - min)/25
    yvar = np.arange(min, max+step, step)
else:
    if ylabel == "ReH":
        yvar = np.arange(-5, 31, 1)
    elif ylabel == "ReE":
        yvar = np.arange(-71, -35, 1)
    else:
        yvar = np.arange(-11, 25, 1)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_zlabel("F")

X, Y = np.meshgrid(xvar, yvar)
df = df[df["#Set"] == set]
f = np.vectorize(function)
Z = f(X, Y)

ax.plot_surface(X, Y, Z)
plt.show()