import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from BHDVCStf import BHDVCS

def to_arrays(phi, k, QQ, xB, t, F1, F2, dvcs, ReH, ReE, ReHtilde, n):
    k = np.full([n, 1], k)
    QQ = np.full([n, 1], QQ)
    xB = np.full([n, 1], xB)
    t = np.full([n, 1], t)
    F1 = np.full([n, 1], F1)
    F2 = np.full([n, 1], F2)
    dvcs = np.full([n, 1], dvcs)
    x = np.concatenate((phi.T, k, QQ, xB, t, F1, F2, dvcs), axis=1)
    ReH = np.full(n, ReH)
    ReE = np.full(n, ReE)
    ReHtilde = np.full(n, ReHtilde)
    return x, ReH, ReE, ReHtilde

# The parametrized function to be plotted
bhdvcs = BHDVCS()
phi_x = np.arange(0, 351, 10)
n = len(phi_x)
phi_x = np.array([phi_x])

# Define initial parameters
i_k = 3.75
i_QQ = 1.74013
i_xB = 0.345012
i_t = -0.38087
i_F1 = 0.49806
i_F2 = 0.68579
i_dvcs = 0.012288
i_ReH = 6.99014
i_ReE = -53.0554
i_ReHtilde = 3.88341

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
x, ReH, ReE, ReHtilde = to_arrays(phi_x, i_k, i_QQ, i_xB, i_t, i_F1, i_F2, i_dvcs, i_ReH, i_ReE, i_ReHtilde, n)
line = plt.scatter(phi_x, bhdvcs.TotalUUXS(x, ReH, ReE, ReHtilde))
ax.set_xlabel('phi_x')

axcolor = "lightgoldenrodyellow"
ax.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(left=0.21, bottom=0.42)

# Make a horizontal slider to control k
axk = plt.axes([0.25, 0.3, 0.65, 0.02], facecolor=axcolor)
k_slider = Slider(
    ax=axk,
    label='k',
    valmin=3.75,
    valmax=7.75,
    valinit=i_k,
)

# Make a horizontal slider to control QQ
axQQ = plt.axes([0.25, 0.27, 0.65, 0.02], facecolor=axcolor)
QQ_slider = Slider(
    ax=axQQ,
    label="QQ",
    valmin=1.74,
    valmax=2.64,
    valinit=i_QQ,
)

axxB = plt.axes([0.25, 0.24, 0.65, 0.02], facecolor=axcolor)
xB_slider = Slider(
    ax=axxB,
    label="xB",
    valmin=0.34,
    valmax=0.44,
    valinit=i_xB,
)

axt = plt.axes([0.25, 0.21, 0.65, 0.02], facecolor=axcolor)
t_slider = Slider(
    ax=axt,
    label="t",
    valmin=-0.38,
    valmax=-0.27,
    valinit=i_t,
)

axF1 = plt.axes([0.25, 0.18, 0.65, 0.02], facecolor=axcolor)
F1_slider = Slider(
    ax=axF1,
    label="F1",
    valmin=0.49,
    valmax=0.58,
    valinit=i_F1,
)

axF2 = plt.axes([0.25, 0.15, 0.65, 0.02], facecolor=axcolor)
F2_slider = Slider(
    ax=axF2,
    label="F2",
    valmin=0.68,
    valmax=0.86,
    valinit=i_F2,
)

axReH = plt.axes([0.25, 0.12, 0.65, 0.02], facecolor=axcolor)
ReH_slider = Slider(
    ax=axReH,
    label="ReH",
    valmin=6.99,
    valmax=13.1,
    valinit=i_ReH,
)

axReE = plt.axes([0.25, 0.09, 0.65, 0.02], facecolor=axcolor)
ReE_slider = Slider(
    ax=axReE,
    label="ReE",
    valmin=-53.1,
    valmax=-46.99,
    valinit=i_ReE,
)

axReHtilde = plt.axes([0.25, 0.06, 0.65, 0.02], facecolor=axcolor)
ReHtilde_slider = Slider(
    ax=axReHtilde,
    label="ReHtilde",
    valmin=3.88,
    valmax=7.26,
    valinit=i_ReHtilde,
)

# The function to be called anytime a slider's value changes
def update(val):
    x, ReH, ReE, ReHtilde = to_arrays(phi_x, k_slider.val, QQ_slider.val, xB_slider.val, t_slider.val, F1_slider.val, F2_slider.val, i_dvcs, ReH_slider.val, ReE_slider.val, ReHtilde_slider.val, n)
    y_vals = np.array([bhdvcs.TotalUUXS(x, ReH, ReE, ReHtilde).numpy()]).T
    line.set_offsets(np.concatenate((phi_x.T, y_vals), axis=1))
    fig.canvas.draw_idle()


# register the update function with each slider
k_slider.on_changed(update)
QQ_slider.on_changed(update)
xB_slider.on_changed(update)
t_slider.on_changed(update)
F1_slider.on_changed(update)
F2_slider.on_changed(update)
ReH_slider.on_changed(update)
ReE_slider.on_changed(update)
ReHtilde_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.8, 0.01, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    k_slider.reset()
    QQ_slider.reset()
    xB_slider.reset()
    t_slider.reset()
    F1_slider.reset()
    F2_slider.reset()
    ReH_slider.reset()
    ReE_slider.reset()
    ReHtilde_slider.reset()
button.on_clicked(reset)

plt.show()
