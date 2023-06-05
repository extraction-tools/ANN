from PyQt5.QtGui import QIntValidator, QDoubleValidator, QRegExpValidator
from PyQt5.QtCore import QThread, pyqtSignal, Qt
import random
import os.path
import datetime
from dateutil.parser import parse
import json
import pytz
from scipy import optimize
import numpy as np
from scipy import interpolate
import cmath
import matplotlib.pyplot as plt
import statistics as std


class Config():

    ''' Contains Relevant Parameter Data for NMR Simulation
    Arguments:
        U: Voltage
        Cknob: Capacitance
        Cable: Cable length
        Eta: Filling factor
        Phi: Const
        Cstray: stray capacitance

        k_range: sweep range
        circ_constants: circuitry constant tuple
        f_input: frequency median

        scansize: sweep size
        ranger: range covered

        Backgmd: background simulation data
        Backreal: real background part simulation
        Current: baseline current simulation

        Other constants: g, s, bigy
    '''

    def __init__(self, params, k_range, f_input, scansize, ranger, backgmd, backreal, current):

        circ_consts = (3*10**(-8),0.35,619,50,10,0.0343,4.752*10**(-9),50,1.027*10**(-10),2.542*10**(-7),0,0,0,0)

        self.L0 = circ_consts[0]
        self.Rcoil = circ_consts[1]
        self.R = circ_consts[2]
        self.R1 = circ_consts[3]
        self.r = circ_consts[4]
        self.alpha = circ_consts[5]
        self.beta1 = circ_consts[6]
        self.Z_cable = circ_consts[7]
        self.D = circ_consts[8]
        self.M = circ_consts[9]
        self.delta_C = circ_consts[10]
        self.delta_phi = circ_consts[11]
        self.delta_phase = circ_consts[12]
        self.delta_l = circ_consts[13]

        self.f = f_input
    
        self.U = params[0]
        self.knob = params[1]
        self.trim = params[2]
        self.eta = params[3]
        self.phi_const = params[4]
        self.Cstray = params[5]

        self.g = 0.05
        self.s = 0.04
        self.bigy=(3-self.s)**0.5

        self.scansize = scansize
        self.k_range = k_range
        self.rangesize = ranger

        # pi = np.pi
        # im_unit = complex(0,1)
        # sign = 1

        # I = self.U*1000/self.R #Ideal constant current, mA

        # #Derived quantities
        # w_res = 2*pi*self.f
        # w_low = 2 * pi * (213 - self.scansize) * (1e6)
        # w_high = 2 * pi * (213 + self.scansize) * (1e6)
        # delta_w = 2 * pi * 500 * ((1e3)/500)

class Simulate():

    def __init__(self,config):
        self.Inputs = config()

    def LabviewCalculateXArray(self):
        
        #---------------------preamble----------------
        
        pi = np.pi
        im_unit = complex(0,1)

        #----------------------main------------------
        
        
        

        #Derived quantities
        w_res = 2*pi*self.Inputs.f
        f_small = self.Inputs.f/(1e6)
        w_low = 2 * pi * (f_small - self.Inputs.scansize) * (1e6)
        w_high = 2 * pi * (f_small + self.Inputs.scansize) * (1e6)
        delta_w = 2 * pi * 500 * ((1e3)/500)

        
        #Variables for creating splines
        k_ints = range(0,500)
        k = np.array(k_ints,float)
        x = (k*delta_w)+(w_low)
        
        larger_k = range(0,self.Inputs.k_range)
        larger_x = np.array(larger_k, float)
        w_range = w_high - w_low
        larger_range = (delta_w*larger_x)+(w_low-5*w_range)
        larger_range /= (2 * pi)*(1e6)
        
        x /= (2*pi)*(1e6)
        return_val = x
        if (self.Inputs.rangesize == 1):
            return_val = larger_range
        return return_val

    def getArrayFromFunc(self,func,inputs):
        output = []
        for input in inputs:
            output.append((func(input)).real)
        return output
    
    def LabviewCalculateYArray(self):
        
        #---------------------preamble----------------
        
        pi = np.pi
        im_unit = complex(0,1)
        sign = 1
        
        #----------------------main------------------

        L0 = self.Inputs.L0
        Rcoil = self.Inputs.Rcoil
        R = self.Inputs.R
        R1 = self.Inputs.R1
        r = self.Inputs.r
        alpha = self.Inputs.alpha
        beta1 = self.Inputs.beta1
        Z_cable = self.Inputs.Z_cable
        D = self.Inputs.D
        M = self.Inputs.M
        delta_C = self.Inputs.delta_C
        delta_phi = self.Inputs.delta_phi
        delta_phase = self.Inputs.delta_phase
        delta_l = self.Inputs.delta_l
        
        
        f = self.Inputs.f
        
        U = self.Inputs.U
        knob = self.Inputs.knob
        trim = self.Inputs.trim
        eta = self.Inputs.eta
        phi_const = self.Inputs.phi_const
        Cstray = self.Inputs.Cstray
        
        I = U*1000/R #Ideal constant current, mA

        #Derived quantities
        w_res = 2*pi*f
        w_low = 2 * pi * (213 - self.Inputs.scansize) * (1e6)
        w_high = 2 * pi * (213 + self.Inputs.scansize) * (1e6)
        delta_w = 2 * pi * 500 * ((1e3)/500)
        
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
            return 1
    
        #More derived quantities
        vel = 1/beta(1)

        #More functions
        def lam(w):
            return vel/f
        
        #Even more derived quantities
        l_const = trim*lam(w_res)

        #Even more functions
        def l(w):
            return l_const + delta_l
            
    
            
        
        #Variables for creating splines
        k_ints = range(0,500)
        k = np.array(k_ints,float)
        x = (k*delta_w)+(w_low)
        Icoil_TE = 0.11133
        
        butxi = []
        butxii = []
        vback = []
        vreal = []    
        Icoil = []
        
        for item in deriv_sig:
            butxi.append(item)
        for item in main_sig:
            butxii.append(item)
        for item in backgmd_sig:
            vback.append(item)
        for item in backreal_sig:
            vreal.append(item)
        for item in current_sig:
            Icoil.append(item)
        
        x1 = interpolate.interp1d(x,butxi,fill_value=0.0,bounds_error=False)
        x2 = interpolate.interp1d(x,butxii,fill_value=0.0,bounds_error=False)
        # b = interpolate.interp1d(x,vback,fill_value="extrapolate",kind="quadratic",bounds_error=False)
        # rb = interpolate.interp1d(x,vreal,fill_value="extrapolate",kind="quadratic",bounds_error=False)
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
