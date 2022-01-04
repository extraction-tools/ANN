from iminuit import Minuit

import numpy as np
import pandas as pd 
from BHDVCS_fit import DVCSFIT

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import chisquare
from statistics import median

dv = DVCSFIT()
func_fit = dv.curve_fit2
f = dv.plot_fit

dats = pd.read_csv('jlab_data_hallB.csv')

k = np.array(dats['#k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF = np.array(dats['sigmaF'])
F1 = np.array(dats['F1']) 
F2 = np.array(dats['F2'])

df = pd.read_csv("jlab_data_hallB.csv")
def F2VsPhi(dataframe,SetNum,xdat,cffs):
    TempFvalSilces=dataframe[(SetNum-1)*24:(SetNum-1)*24+24]
    TempFvals=TempFvalSilces["F"]
    TempFvals_sigma=TempFvalSilces["sigmaF"]
    mask = (TempFvals_sigma[0:24] > 0)
    temp_phi=TempFvalSilces["phi_x"]
    plt.errorbar(temp_phi[mask],TempFvals[mask],TempFvals_sigma[mask],fmt='.',color='blue',label="Data")
    plt.xlim(0,368)
    temp_unit=(np.max(TempFvals)-np.min(TempFvals))/len(TempFvals)
    plt.ylim(np.min(TempFvals)-temp_unit,np.max(TempFvals)+temp_unit)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc=4,fontsize=10,handlelength=3)
    plt.title("Local fit with data set #"+str(SetNum),fontsize=20)
    plt.plot(temp_phi[mask], f(xdat, cffs), 'g--', label='fit')
    file_name = "plot_set_number_{}.png".format(SetNum)
    plt.savefig(file_name)


def LSQ(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.sum((ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ** 2 / errs ** 2)

def MLM(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.sum(func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit) - ydat + ydat * np.log(ydat/func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)))

def chisq(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.mean((ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ** 2 / errs ** 2)

def MAE(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.mean(np.abs(ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit))/ydat)

for ii in range(10):
 a = 24*ii # start index of set
 b = a+24 # end index of set
 
 mask = (errF[a:b] > 0)
 xdat = np.array([phi[a:b][mask], qq[a:b][mask], xb[a:b][mask], t[a:b][mask], k[a:b][mask], F1[a:b][mask], F2[a:b][mask]])
 ydat = np.array([F[a:b][mask]])
 errs = np.array([errF[a:b][mask]])

 m = Minuit(LSQ, ReH = 0.0001, ReE = 0.0001, ReHtilde = 0.0001, c0fit = 0.0001, c1fit = 0.) # choose LSQ or MLM
 ##m.limits = [(-10, 10), (-10, 10), (-10, 10), (0, 10), (0, 10)] # parm limit not sure if this is useful
 m.errordef = Minuit.LEAST_SQUARES #important for the correct fit error but careful with this
 #m.errordef = Minuit.LIKELIHOOD #important for the correct fit error but careful with this
 ##m.errors = (0.00000001, 0.00000001,0.00000001,0.00000001,0.00000001) #initial step size not sure if this is useful

 ## this is to fix the parameter. Fix the last parameter if you use twist 2
 #m.fixed[0] = True
 #m.fixed[1] = True
 #m.fixed[2] = True
 #m.fixed[3] = True
 m.fixed[4] = True
 m.migrad()
 m.hesse()

 ReH_fit = m.values["ReH"]
 ReE_fit = m.values["ReE"]
 ReHtilde_fit = m.values["ReHtilde"]
 c0fit_fit = m.values["c0fit"]
 c1fit_fit = m.values["c1fit"] 
 cffs = [ReH_fit, ReE_fit, ReHtilde_fit, c0fit_fit, c1fit_fit]
 chisq_val = chisq(ReH_fit, ReE_fit, ReHtilde_fit, c0fit_fit, c1fit_fit) 
 MAE_val = MAE(ReH_fit, ReE_fit, ReHtilde_fit, c0fit_fit, c1fit_fit)
 
 print('%.4f %.4f %.4f %.4f %.4f' % (ReH_fit, ReE_fit, ReHtilde_fit, chisq_val, MAE_val))
 F2VsPhi(df,ii+1,xdat,cffs)
 plt.clf()  

