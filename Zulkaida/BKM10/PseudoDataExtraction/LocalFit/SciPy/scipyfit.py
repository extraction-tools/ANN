import torch
import torch.nn as nn 
from torch.autograd import Variable 
import torch.nn.functional as F 
import torch.utils.data as Data

import numpy as np
import pandas as pd 
from BHDVCS_torch import TBHDVCS
from BHDVCS_fit import DVCSFIT


import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.stats import chisquare
from statistics import median


tb = TBHDVCS()
f = tb.curve_fit

dv = DVCSFIT()
func_fit = dv.curve_fit2

loss_validation1 = tb.loss_MSE
loss_validation2 = tb.loss_chisq
loss_validation3 = tb.loss_MAE

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

for ii in range(5):
 a = 24*ii # start index of set
 b = a+24 # end index of set
 
 mask = (errF[a:b] > 0)
 xdat = np.array([phi[a:b][mask], qq[a:b][mask], xb[a:b][mask], t[a:b][mask], k[a:b][mask], F1[a:b][mask], F2[a:b][mask]])
 ydat = np.array([F[a:b][mask]])

 
 y = Variable(torch.from_numpy(ydat.transpose()))
 xdat = Variable(torch.from_numpy(xdat))
 errs = Variable(torch.from_numpy(errF[a:b][mask]))
 
 popt = fit_scipy(ii)
 cffs = [popt[0], popt[1], popt[2], popt[3], popt[4]]  
 loss_val1 = loss_validation1((xdat.float()), cffs, errs, y)
 loss_val2 = loss_validation2((xdat.float()), cffs, errs, y)
 loss_val3 = loss_validation3((xdat.float()), cffs, errs, y)

 print('%.4f %.4f %.4f %.8f %.4f %.4f' % (popt[0], popt[1], popt[2], loss_val1, loss_val2, loss_val3 ))
 F2VsPhi(df,ii+1,xdat,cffs)
 plt.clf()  

