import numpy as np
from scipy import interpolate
from scipy import stats
from ROOT import gROOT, TCanvas, TGraph, gApplication, TF1
import array as arr
import cmath
import sys

args = sys.argv

def getGraphFromFunc(f, inputs):
    outputs = arr.array('d')
    inp = arr.array('d')
    for i in inputs:
        inp.append(i)
        outputs.append(f(i))
    graph = TGraph(len(inputs),inp,outputs)
    return graph

def getGraphFromArrays(n,x,y):
    inputs = arr.array('d')
    outputs = arr.array('d')
    for i in range(len(x)):
        inputs.append(x[i])
        outputs.append(y[i])
    graph = TGraph(len(inputs),inputs,outputs)
    return graph

#---------------------preamble----------------

pi = np.pi
im_unit = complex(0,1)
sign = -1
eta = 0.00001*2.6*400

#----------------------main------------------

#Proton resonant frequency at 214 MHz, with 7 l/2 cable

gROOT.Reset()

#Parameters of problem
U = 0.1 #100 mV RF input

#Coil
L0 = 3e-8 #Inductance of coil 30 nH
Rcoil = 0.35  #Resistance of coil 0.35 ohm

R = 619 #Constant current resistance
R1 = 50 #Amp impedance 50 ohm
r = 10 #Damping resistor
 
f = 213e6 #213 MHz scan
knob = 0.885 #knob of tuning capacitor
if (len(args) > 1):
    knob = float(args[1])
Cstray = 1e-15 #Stray capacitance 10^-3 pF
trim = 7*0.5
phi_const = 6.1319
delta_C = 0
delta_phi = 0
delta_phase = 0
I = U*1000/R #Ideal constant current, mA

#Derived quantities
w_res = 2*pi*f
w_low = 2 * pi * (213 - 0.25) * (1e6)
w_high = 2 * pi * (213 + 0.25) * (1e6)
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

#Parameters
alpha = 0.0343 #estimate from August 1997
beta1 = 4.752e-9
Z_cable = 50 #Cable impedance 50 ohms
D = 10.27e-11 #F/m
M = 2.542e-7
delta_l = 0

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
    temp_variable = complex(0,w*C(w))
    return 1/temp_variable

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





#------------------READING IN DATA FROM FILES-------------

#Variables for creating splines
k_ints = range(0,256)
k = np.array(k_ints,float)
x = (k*delta_w)+(w_low)
Icoil_TE = 0.11133

butxi = []
butxii = []
vback = []
vreal = []

Icoil = []

f = open("data/DPROTON.DAT","r") #DPROTON.DAT
for line in f:
    butxi.append(float(line))
f.close()

f2 = open("data/PROTON.DAT","r") #PROTON.DAT
for line in f2:
    butxii.append(float(line))
f2.close()

f3 = open("data/Backgmd.dat","r")
for line in f3:
    vback.append(float(line))
f3.close()

f4 = open("data/Backreal.dat","r")
for line in f4:
    vreal.append(float(line))
f4.close()

f5 = open("data/current.dat","r")
for line in f5:
    Icoil.append(float(line))
f5.close()

x1 = interpolate.interp1d(x,butxi,fill_value="extrapolate")
x2 = interpolate.interp1d(x,butxii,fill_value="extrapolate")
b = interpolate.interp1d(x,vback,fill_value="extrapolate")
rb = interpolate.interp1d(x,vreal,fill_value="extrapolate")

#Chi is the magnetic susceptibility. x1 is the dispersive term, and x2 is the absorptive term
def chi(w):
    return complex(x1(w),-1*x2(w))

#print("w_low: " + str(w_low))
#print("x[0]: " + str(x[0]))
#print("w_high: " + str(w_high))
#print("x[1]: " + str(x[1]))

ic = interpolate.interp1d(x,Icoil,fill_value="extrapolate")

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
    return 1/(im_unit*w*Cstray)

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

print("a = "+str(a))
print("b = "+str(b))
print("c = "+str(c))


def parfaze(w):
    return a*w*w + bb*w + c

def phi_trim(w):
    return slope_phi()*(w-w_res) + parfaze(w)

def phi(w):
    return phi_trim(w) + phi_const

def V_out(w):
    return I*Ztotal(w)*np.exp(im_unit*phi(w)*pi/180)


#array_size = 1000
#wk_ints = range(0,array_size)
#wk = np.array(wk_ints, float)
#w = ((wk*delta_w)+(w_low-((array_size/2)*delta_w)))

#print("w_low = "+str(w_low))
#print("w_high = "+str(w_high))
#print("delta_w = "+str(delta_w))

tg3 = getGraphFromFunc(V_out,x)
tg3.SetTitle("Output signal;#omega;Voltage (V)")
tg3.Draw() 

gApplication.Run()
