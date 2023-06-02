# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:39:46 2022

@author: Devin
"""

import numpy as np
from scipy import interpolate
from scipy import stats
import matplotlib.pyplot as plt
import cmath

# ---- Constants ---- #
k_range = 30000
circ_constants = (3*10**(-8),0.35,619,50,10,0.0343,4.752*10**(-9),50,1.027*10**(-10),2.542*10**(-7),0,0,0,0)
circ_params = (0.1,0.2,3.5,0.0104,6.1319,10**(-15))
# circ_params = (0.1,0.885,3.5,0.0104,6.1319,10**(-15))
function_input = 32000000
# function_input = 213000000
# function_input = 107000000
# function_input = 0
scan_s = .25
ranger = 1
# ---- Data Files ---- #
Backgmd = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\Backgmd.dat', unpack = True)
Backreal = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\Backreal.dat', unpack = True)
Current = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\current.dat', unpack = True)
Deuteron_Dat = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\DEUTERON.dat', unpack = True)
Deuteron_Deriv = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\DDEUTERON.dat', unpack = True)
# Deuteron_Dat = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\PROTON.dat', unpack = True)
# Deuteron_Deriv = np.loadtxt(r'C:\Users\Devin\Desktop\Spin Physics Work\ANN-NMR-main\Qmeter_Simulation\qmeter_simulation-master\data\DPROTON.dat', unpack = True)
# Deuteron_Dat = np.zeros(256)
# Deuteron_Deriv = np.zeros(256)

def LabviewCalculateXArray(f_input, scansize, rangesize):
    
    #---------------------preamble----------------
    
    pi = np.pi
    im_unit = complex(0,1)

    #----------------------main------------------
    
    
    

    #Derived quantities
    w_res = 2*pi*f_input
    f_small = f_input/(1e6)
    # f_small = f_input
    w_low = 2 * pi * (f_small - scansize) * (1e6)
    w_high = 2 * pi * (f_small + scansize) * (1e6)
    delta_w = 2 * pi * 500 * ((1e3)/256)

    
    #Variables for creating splines
    k_ints = range(0,256)
    k = np.array(k_ints,float)
    x = (k*delta_w)+(w_low)
    
    larger_k = range(0,k_range)
    larger_x = np.array(larger_k, float)
    w_range = w_high - w_low
    larger_range = (delta_w*larger_x)+(w_low-5*w_range)
    larger_range /= (2 * pi)*(1e6)
    
    x /= (2*pi)*(1e6)
    return_val = x
    if (rangesize == 1):
        return_val = larger_range
    return return_val


def getArrayFromFunc(func,inputs):
    output = []
    for input in inputs:
        output.append((func(input)).real)
    return output

def LabviewCalculateYArray(circ_consts, params, f_input, scansize, main_sig, deriv_sig, backgmd_sig, backreal_sig, current_sig, rangesize):
    
    #---------------------preamble----------------
    
    pi = np.pi
    im_unit = complex(0,1)
    sign = 1
    
    #----------------------main------------------

    L0 = circ_consts[0]
    Rcoil = circ_consts[1]
    R = circ_consts[2]
    R1 = circ_consts[3]
    r = circ_consts[4]
    alpha = circ_consts[5]
    beta1 = circ_consts[6]
    Z_cable = circ_consts[7]
    D = circ_consts[8]
    M = circ_consts[9]
    delta_C = circ_consts[10]
    delta_phi = circ_consts[11]
    delta_phase = circ_consts[12]
    delta_l = circ_consts[13]
    
    
    f = f_input
    
    ### Input Parameters ###
    
    U = params[0]
    knob = params[1]
    trim = params[2]
    eta = params[3]
    phi_const = params[4]
    Cstray = params[5]
    
    I = U*1000/R #Ideal constant current, mA

    #Derived quantitiesL0
    w_res = 2*pi*f
    w_low = 2 * pi * (213 - scansize) * (1e6)
    w_high = 2 * pi * (213 + scansize) * (1e6)
    # w_low = 2 * pi * (213 - scansize)
    # w_high = 2 * pi * (213 + scansize)
    delta_w = 2 * pi * 500 * ((1e3)/256)
    
    #Functions
    def slope():
        return delta_C / (0.25 * 2 * pi * 1e6)

    def slope_phi():
        return delta_phi / (0.25 * 2 * pi * 1e6)

    def Ctrim(w):
        return slope()*(w - w_res)

    def Cmain():
        return 20*(1e-12)*knob

    def C(w):
        return Cmain() + Ctrim(w)*(1e-12)

    def Cpf():
        return C(w_res)*(1e12)
    
    
    #--------------------Cable characteristics-------------


    #Derived quantities
    S = 2*Z_cable*alpha

    #Functions

    def Z0(w):
        return cmath.sqrt( (S + w*M*im_unit) / (w*D*im_unit))

    def beta(w):
        return beta1*w

    def gamma(w):
        return complex(alpha,beta(w))

    def ZC(w):
        if  w != 0 and C(w) != 0:
            return 1/(im_unit*w*C(w))
        else:
            return 1
  
    #More derived quantities
    vel = 1/beta(1)

    # More functions
    def lam(f):
        return vel/f
    # def lam(w):
    #     return vel/w
    
    
    # Even more derived quantities
    l_const = trim*lam(f)

    #Even more functions
    def l(w):
        return l_const + delta_l
        
        
    
    #Variables for creating splines
    k_ints = range(0,256)
    k = np.array(k_ints,float)
    x = (k*delta_w)+(w_low)
    Icoil_TE = 0.11133
    
    
    butxi = np.array([])
    butxii = np.array([])
    vback = np.array([])
    vreal = np.array([]) 
    Icoil = np.array([])
    

    for item in deriv_sig:
        butxi = np.append(butxi,item)
    for item in main_sig:
        butxii = np.append(butxii,item)
    for item in backgmd_sig:
        vback = np.append(vback,item)
    for item in backreal_sig:
        vreal = np.append(vreal,item)
    for item in current_sig:
        Icoil = np.append(Icoil,item)
        

    
    
    x1 = interpolate.interp1d(x,butxi,fill_value="extrapolate",kind="quadratic",bounds_error=False)
    x2 = interpolate.interp1d(x,butxii,fill_value="extrapolate",kind="quadratic",bounds_error=False)
    b = interpolate.interp1d(x,vback,fill_value="extrapolate",kind="quadratic",bounds_error=False)
    rb = interpolate.interp1d(x,vreal,fill_value="extrapolate",kind="quadratic",bounds_error=False)
    ic = interpolate.interp1d(x,Icoil,fill_value="extrapolate",kind="linear",bounds_error=False)
    
    
    
    def chi(w):
        return complex(x1(w),-1*x2(w))

    def pt(w):
        return ic(w)/Icoil_TE

    def L(w):
        return L0*(1+(sign*4*pi*eta*pt(w)*chi(w)))

    def real_L(w):
        return L(w).real

    def imag_L(w):
        return L(w).imag

    def ZLpure(w):
        return im_unit*w*L(w) + Rcoil

    def Zstray(w):
        if w != 0 and Cstray !=0:
            return 1/(im_unit*w*Cstray)
        return 1

    def ZL(w):
        return ZLpure(w)*Zstray(w)/(ZLpure(w)+Zstray(w))

    def ZT(w):
        return Z0(w)*(ZL(w) + Z0(w)*np.tanh(gamma(w)*l(w)))/(Z0(w) + ZL(w)*np.tanh(gamma(w)*l(w)))


    def Zleg1(w):
        return r + ZC(w) + ZT(w)

    def Ztotal(w):
        return R1 / (1 + (R1 / Zleg1(w)) )

    #Adding parabolic term
    # for w in range (0,100):
    #     print(x1(w))

    xp1 = w_low
    xp2 = w_res
    xp3 = w_high
    yp1 = 0
    yp2 = delta_phase
    yp3 = 0

    alpha_1 = yp1-yp2
    alpha_2 = yp1-yp3
    beta_1 = (xp1*xp1) - (xp2*xp2)
    beta_2 = (xp1*xp1) - (xp3*xp3)
    gamma_1 = xp1-xp2
    gamma_2 = xp1-xp3
    temp=(beta_1*(gamma_1/gamma_2) - beta_2)
    a= (gamma_2 *(alpha_1/gamma_1) - alpha_2)/temp
    bb = (alpha_2 - a*beta_2)/gamma_2
    c = yp1 - a*xp1*xp1 - bb*xp1

    def parfaze(w):
        return a*w*w + bb*w + c

    def phi_trim(w):
        return slope_phi()*(w-w_res) + parfaze(w)

    def phi(w):
        return phi_trim(w) + phi_const

    def V_out(w):
        return -1*(I*Ztotal(w)*np.exp(im_unit*phi(w)*pi/180))

    
    larger_k = range(0,k_range)
    larger_x = np.array(larger_k, float)
    w_range = w_high - w_low
    larger_range = (delta_w*larger_x)+(w_low-5*w_range)
    
    out_y = getArrayFromFunc(V_out,x)
    if (rangesize == 1):
        out_y = getArrayFromFunc(V_out,larger_range)
    return out_y


y = LabviewCalculateYArray(circ_constants, circ_params, function_input, scan_s, Deuteron_Dat, Deuteron_Deriv, Backgmd, Backreal, Current, ranger)
x = LabviewCalculateXArray(function_input, scan_s, ranger)
plt.plot(x,y,"black",linewidth=1)
plt.xlabel("Frequency")
plt.ylabel("Voltage")
# plt.xlim([57.8,60.2])
# plt.savefig('Q_Meter_Proton.png')