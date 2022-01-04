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

dats = pd.read_csv('pseudo_BKM10_hallB_t3.csv')

ind = np.array(dats['index'])
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF = np.array(dats['sigmaF'])
F1 = np.array(dats['F1']) 
F2 = np.array(dats['F2'])
gReH = np.array(dats['gReH'])
gReE = np.array(dats['gReE'])
gReHTilde = np.array(dats['gReHTilde'])


df = pd.read_csv("pseudo_BKM10_hallB_t3.csv")
def F2VsPhi(dataframe,SetNum,xdat,cffs):
    TempFvalSilces=dataframe[dataframe["#Set"]==SetNum]
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

def fit_scipy(n):
    
    a = (np.amax(ind)+1)*n
    b = a + np.amax(ind)
    mask = (errF[a:b] > 0)
    xdat = (phi[a:b][mask], qq[a:b][mask], xb[a:b][mask], t[a:b][mask], k[a:b][mask], F1[a:b][mask], F2[a:b][mask])
    popt, pcov = curve_fit(func_fit, xdat, F[a:b][mask], sigma=errF[a:b][mask],  method='trf')
    ##popt, pcov = curve_fit(func_fit, xdat, F[a:b][mask], sigma=errF[a:b][mask], method='trf', bounds = (-100.,100))
    return popt[0], popt[1], popt[2], popt[3], popt[4]

def chisq(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.mean((ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ** 2 / errs ** 2)
def MAE(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.mean(np.abs(ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit))/ydat)

for ii in range(5):
 a = 24*ii # start index of set
 b = a+24 # end index of set
 
 mask = (errF[a:b] > 0)
 xdat = np.array([phi[a:b][mask], qq[a:b][mask], xb[a:b][mask], t[a:b][mask], k[a:b][mask], F1[a:b][mask], F2[a:b][mask]])
 ydat = np.array([F[a:b][mask]])
 errs = np.array([errF[a:b][mask]])
 
 popt = fit_scipy(ii)
 cffs = [popt[0], popt[1], popt[2], popt[3], popt[4]]  
 chisq_val = chisq(popt[0], popt[1], popt[2], popt[3], popt[4])
 MAE_val = MAE(popt[0], popt[1], popt[2], popt[3], popt[4])

 print('%.4f %.4f %.4f %.8f %.4f' % (popt[0], popt[1], popt[2], chisq_val, MAE_val ))
 F2VsPhi(df,ii+1,xdat,cffs)
 plt.clf()  

