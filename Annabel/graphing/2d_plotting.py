from TVA1_UU import TVA1_UU #modified bhdvcs file
from BHDVCStf import BHDVCS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
creates a 2d scatterplot with [k, QQ, x_b, t, ReH, ReE, ReHtilde, or dvcs]
on the x-axis and the integral of F w/ respect to phi on the y-axis

 * constant values of kinematics, CFF's, etc. are from any #Set in datafile
 * range of values for x-axis depend on the range in datafile
 * (or, if unavailable, hard coded for CFF's)

FIELDS -
line 23: #Set in datafile
line 24: formulation, TVA1_UU vs. BHDVCS
line 27: datafile name
line 33: step size of x-axis
"""

# setup
set = 1
tv = TVA1_UU() # tv = BHDVCS()
xlabel = input("Enter variable on x-axis (k, QQ, x_b, t, ReH, ReE, ReHtilde, dvcs): ")
xlabel.rstrip()
df = pd.read_csv("dvcs_May21.csv") # df = pd.read_csv("dvcs_xs_newsets_genCFFs.csv")

# create range of values for independent variable
if xlabel in df:
    min = np.amin(df[xlabel])
    max = np.amax(df[xlabel])
    step = (max - min)/20
    var = np.arange(min, max+step, step)
else:
    if xlabel == "ReH":
        var = np.arange(-5, 31, 1)
    elif xlabel == "ReE":
        var = np.arange(-71, -35, 1)
    else:
        var = np.arange(-11, 25, 1)

# initialize constant numpy arrays for F(input)
df = df[df["#Set"] == set]
inputs = dict()
inputs["phi_x"] = np.array(df[["phi_x"]])
n = len(inputs["phi_x"])
inputs["ReH"] = np.full([n, 1], 9.6)
inputs["ReE"] = np.full([n, 1], -49.6)
inputs["ReHtilde"] = np.full([n, 1], 5.3)
for label in ["k", "QQ", "x_b", "t", "F1", "F2", "dvcs", "ReH", "ReE", "ReHtilde"]:
    if label in df:
        inputs[label] = np.array(df[[label]])

# iterate through IV range
x_points = []
y_points = []
for x in var:
    x_points.append(x)
    inputs[xlabel] = np.full([n, 1], x)
    kinematics = np.concatenate((inputs["phi_x"], inputs["k"], inputs["QQ"], inputs["x_b"], inputs["t"], inputs["F1"], inputs["F2"], inputs["dvcs"]), axis=1)
    F = tv.TotalUUXS(kinematics, inputs["ReH"], inputs["ReE"], inputs["ReHtilde"])
    area = 0
    phi_step = inputs["phi_x"][1, 0] - inputs["phi_x"][0, 0]
    for val in F.numpy()[0, :]:
        area = area + val * phi_step
    y_points.append(area)

plt.scatter(x_points, y_points)
plt.xlabel(xlabel)
plt.ylabel("integral of F w/ respect to phi_x")
plt.show()