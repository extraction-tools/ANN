import numpy as np
import pandas as pd
from BHDVCS_fit import DVCSFIT

dv = DVCSFIT()
func_fit = dv.curve_fit2
f = dv.plot_fit
F1_term = dv.Get_F1
F2_term = dv.Get_F2

#input kinematics
arr_k = np.array([5.8, 5.8, 5.8, 5.8])
arr_qq = np.array([1.82, 1.93, 1.96, 1.98])
arr_xb = np.array([0.34, 0.36, 0.37, 0.39])
arr_t = np.array([-0.17, -0.23, -0.27, -0.32])

n_phi = 45 # number of phi in one set


Set = np.array([])
ind = np.array([])
phi = np.array([])
k = np.array([])
qq = np.array([])
xb = np.array([])
t = np.array([])
gReH = np.array([])
gReE = np.array([])
gReHTilde = np.array([])
gdvcs = np.array([])

F1 = np.array([])
F2 = np.array([])


for ii in range(arr_k.size):
  for jj in range(n_phi):
    Set = np.append(Set, ii+1)
    ind = np.append(ind,jj)
    phi = np.append(phi, 360.0/n_phi*jj + 0.5 * 360/n_phi)
    k = np.append(k, arr_k[ii])
    qq = np.append(qq, arr_qq[ii])
    xb = np.append(xb, arr_xb[ii])
    t = np.append(t, arr_t[ii])
    
F1 = F1_term(t)
F2 = F2_term(t)
#Input model and smearing error
gReH = 5*t*t + 2*xb*xb
gReE =  4.5*t - 1.5*xb*xb
gReHTilde = -5.5*t + 4.5*xb
gdvcs = xb/10.

smear_err = 0.05 #percentage error

#Get F and errir
f_xdat = np.array([phi, qq, xb, t, k, F1, F2])
f_truth = func_fit(f_xdat, gReH, gReE, gReHTilde, gdvcs)

F_replica = np.random.normal(f_truth, smear_err*f_truth)
F = F_replica
err_smear =  smear_err * f_truth

dat = np.array([Set, ind, k, qq, xb, t, phi, F, err_smear,  F1, F2, gReH, gReE, gReHTilde, gdvcs])
dat = dat.T
np.savetxt('pseudodata_generation.csv', dat, delimiter = ',', fmt='%.7f')
