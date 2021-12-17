import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from TVA1_UU import TVA1_UU
from BHDVCStf import BHDVCS

"""
creates a 3D scatter plot with [k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs]
on the x-, y- and z-axis, while the color varies according to 
the integral of F w/ respect to phi 

 * constant values of kinematics, CFFs, etc. are from any #Set in datafile (in legend)
 * range of values for x-, y- and z-axis depend on the range in datafile
 * (or, if unavailable, hard coded for CFFs)

FIELDS -
line 51: #Set in datafile
line 52: datafile name
line 53: formulation, TVA1_UU vs. BHDVCS
line 60: step size of x-axis
line 74: step size of y-axis
"""

def function(x, y, z):
    # x, y and z are the varying inputs
    inputs[xlabel] = np.full([n, 1], x)
    inputs[ylabel] = np.full([n, 1], y)
    inputs[zlabel] = np.full([n, 1], z)

    kinematics = np.concatenate((phi_x, inputs["k"], inputs["QQ"], inputs["x_b"], inputs["t"], inputs["F1"], inputs["F2"], inputs["dvcs"]), axis=1)
    F = tv.TotalUUXS(kinematics, inputs["ReH"], inputs["ReE"], inputs["ReHtilde"])
    area = 0.0
    for val in F.numpy()[0, :]:
        area = area + (val * phi_step)
    return area

xlabel = input("Enter variable on x-axis (k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs): ")
ylabel = input("Enter variable on y-axis (k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs): ")
zlabel = input("Enter variable on z-axis (k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs): ")
xlabel.strip()
ylabel.strip()
zlabel.strip()

set = 1
df = pd.read_csv("dvcs_May21.csv")
tv = TVA1_UU()
ax = plt.axes(projection='3d')
axis_steps = 7
CFF_step = 5

# range of x axis
if xlabel in df:
    min = np.amin(df[xlabel])
    max = np.amax(df[xlabel])
    step = (max - min)/axis_steps
    xvar = np.arange(min, max+step, step)
else:
    if xlabel == "ReH":
        xvar = np.arange(-5, 31, CFF_step)
    elif xlabel == "ReE":
        xvar = np.arange(-71, -35, CFF_step)
    else:
        xvar = np.arange(-11, 25, CFF_step)

# range of y axis
if ylabel in df:
    min = np.amin(df[ylabel])
    max = np.amax(df[ylabel])
    step = (max - min)/axis_steps
    yvar = np.arange(min, max+step, step)
else:
    if ylabel == "ReH":
        yvar = np.arange(-5, 31, CFF_step)
    elif ylabel == "ReE":
        yvar = np.arange(-71, -35, CFF_step)
    else:
        yvar = np.arange(-11, 25, CFF_step)

# range of z axis
if zlabel in df:
    min = np.amin(df[zlabel])
    max = np.amax(df[zlabel])
    step = (max - min)/axis_steps
    zvar = np.arange(min, max+step, step)
else:
    if zlabel == "ReH":
        zvar = np.arange(-5, 31, CFF_step)
    elif zlabel == "ReE":
        zvar = np.arange(-71, -35, CFF_step)
    else:
        zvar = np.arange(-11, 25, CFF_step)

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_zlabel(zlabel)

df = df[df["#Set"] == set]
phi_x = np.array(df[["phi_x"]])
n = len(phi_x)
inputs = dict()
inputs["ReH"] = np.full([n, 1], 9.6)
inputs["ReE"] = np.full([n, 1], -49.6)
inputs["ReHtilde"] = np.full([n, 1], 5.3)
for label in ["k", "QQ", "x_b", "t", "F1", "F2", "dvcs", "ReH", "ReE", "ReHtilde"]:
    if label in df:
        inputs[label] = np.array(df[[label]])
phi_step = phi_x[1, 0] - phi_x[0, 0]

X, Y, Z = np.meshgrid(xvar, yvar, zvar)
f = np.vectorize(function)
colors = f(X, Y, Z)

img = ax.scatter(X, Y, Z, s=10, c=colors, cmap=plt.winter())
cbar = plt.colorbar(img, pad=0.2)
cbar.set_label('Integral of F')
consts = []
for key in inputs:
    if key == xlabel or key == ylabel or key == zlabel or key == "F1" or key == "F2":
        continue
    consts.append(key + " = " + str(inputs[key][0, 0]))
extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
ax.legend([extra]*6, ("constants taken from set " + str(set), consts[0], consts[1], consts[2], consts[3], consts[4]), bbox_to_anchor=(0.17, 1.13), fontsize='xx-small')
plt.show()