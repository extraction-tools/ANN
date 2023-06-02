# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 09:43:26 2022

@author: Devin
"""

import numpy as np
import matplotlib.pyplot as plt

from qmeter_labview_get_domain import *
from qmeter_labview_node import *
# from qmeter_labview_node_old import *

Backgmd = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\Backgmd.dat', unpack = True)
Backreal = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\Backreal.dat', unpack = True)
Current = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\current.dat', unpack = True)
# Deuteron_Dat = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\DEUTERON.dat', unpack = True)
# Deuteron_Deriv = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\DDEUTERON.dat', unpack = True)
Deuteron_Dat = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\PROTON.dat', unpack = True)
Deuteron_Deriv = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\DPROTON.dat', unpack = True)
# Deuteron_Dat = np.zeros(256)
# Deuteron_Deriv = np.zeros(256)
circ_constants = (3*10**(-8),0.35,619,50,10,0.0343,4.752*10**(-9),50,1.027*10**(-10),2.542*10**(-7),0,0,0,0)
circ_params = (0.1,0.885,3.5,0.0104,6.1319,10**(-15))
# circ_params = (0.1,0.885,3.5,0.0104,6.1319,10**(-15))
# function_input = 32000000
function_input = 213000000
# function_input = 0
scan_s = 0.25
ranger = 1

y = LabviewCalculateYArray(circ_constants, circ_params, function_input, scan_s, Deuteron_Dat, Deuteron_Deriv, Backgmd, Backreal, Current, ranger)

x = LabviewCalculateXArray(function_input, scan_s, ranger)
# print(y)
# print(len(y))
# print(len(x))

# print(len(x))
 # numpy.dot(numpy.ones([97, 2]), numpy.ones([2, 1])).shape
# plt.plot(x,y+(Deuteron_Deriv*Deuteron_Dat))
# plt.xlim([210,216])
# plt.ylim([-100,100])
# plt.ylim()
plt.plot(x,y)
plt.xlabel("Frequency")
plt.ylabel("Voltage")
# plt.savefig('Q_Meter_Curve_Proton.png')

# x = np.linspace(-3,3,256)

# plt.plot(x,Deuteron_Dat)
# plt.plot(x,Deuteron_Deriv)
# plt.plot(x,Backgmd)
# plt.plot(x,Backreal)
# plt.plot(x,Current)
