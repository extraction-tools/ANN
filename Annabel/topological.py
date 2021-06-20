import numpy as np
import matplotlib.pyplot as plt
from TVA1_UU import TVA1_UU

def create_x(phi, k, QQ, xB, t, F1, F2, dvcs, n):
    # input (ints): constant kinematics and n = length of phi_x
    k = np.full([n, 1], k)
    QQ = np.full([n, 1], QQ)
    xB = np.full([n, 1], xB)
    t = np.full([n, 1], t)
    F1 = np.full([n, 1], F1)
    F2 = np.full([n, 1], F2)
    dvcs = np.full([n, 1], dvcs)
    phi = np.array([phi])
    x = np.concatenate((phi.T, k, QQ, xB, t, F1, F2, dvcs), axis=1)
    return x

def function(x, y, xlabel, ylabel):
    # x and y are the varying inputs
    tv = TVA1_UU()
    phi_x = np.arange(8, 361, 8)
    n = len(phi_x)
    inputs = dict()
    inputs["k"] = 2.75
    inputs["QQ"] = 1.516
    inputs["xB"] = 0.3692
    inputs["t"] = -0.307
    inputs["F1"] = 0.543
    inputs["F2"] = 0.799
    inputs["dvcs"] = 0.0122
    inputs["ReH"] = 13
    inputs["ReE"] = -53
    inputs["ReHtilde"] = 7.25
    inputs[xlabel] = x
    inputs[ylabel] = y

    kin = create_x(phi_x, inputs["k"], inputs["QQ"], inputs["xB"], inputs["t"], inputs["F1"], inputs["F2"], inputs["dvcs"], n)
    F = tv.TotalUUXS(kin, np.full(n, inputs["ReH"]), np.full(n, inputs["ReE"]), np.full(n, inputs["ReHtilde"]))
    area = 0
    for val in F.numpy():
        area = area + val * 8
    return area

xlabel = input("Enter variable on x-axis (k, QQ, xB, t, ReH, ReE, ReHtilde): ")
ylabel = input("Enter variable on y-axis: ")
xlabel.strip()
ylabel.strip()
ax = plt.axes(projection='3d')

# ranges of x and y axes for nested for loop
ReH = np.arange(6, 14, 0.5)
ReE = np.arange(-54, -46, 0.5)
ReHtilde = np.arange(3, 9, 0.3)
k = np.arange(2, 25, 1)
QQ = np.arange(1, 15, 0.5)
xB = np.arange(0.1, 0.7, 0.05)
t = np.arange(-0.5, -0.1, 0.025)

vars = dict()
vars["ReH"] = ReH
vars["ReE"] = ReE
vars["ReHtilde"] = ReHtilde
vars["k"] = k
vars["QQ"] = QQ
vars["xB"] = xB
vars["t"] = t

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_zlabel("F")

X, Y = np.meshgrid(vars[xlabel], vars[ylabel])
f = np.vectorize(function)
Z = f(X, Y, xlabel, ylabel)

ax.plot_surface(X, Y, Z)
plt.show()
print("done")