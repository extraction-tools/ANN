import numpy as np
import pandas as pd
from BHDVCS_fit import DVCSFIT

dv = DVCSFIT()
func_fit = dv.curve_fit2
f = dv.plot_fit
F1_term = dv.Get_F1
F2_term = dv.Get_F2
dats = pd.read_csv('pseudo_KM15.csv')
Set = np.array(dats['#Set'])
ind = np.array(dats['index'])
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF = np.array(dats['sigmaF'])
F1 = F1_term(t)
F2 = F2_term(t)
gReH = np.array(dats['ReH'])
gReE = np.array(dats['ReE'])
gReHTilde = np.array(dats['ReHTilde'])
gc0fit = np.array(dats['dvcs'])

#get F
smear_err = 0.05
f_xdat = np.array([phi, qq, xb, t, k, F1, F2])
f_truth = func_fit(f_xdat, gReH, gReE, gReHTilde, gc0fit)

F_replica = np.random.normal(f_truth, smear_err*f_truth)
F = F_replica
err_smear =  smear_err * f_truth

dat = np.array([Set, ind, k, qq, xb, t, phi, F, err_smear,  F1, F2, gReH, gReE, gReHTilde, gc0fit])
#dat = np.array([Set, ind, F, err_smear])
dat = dat.T
#np.savetxt('pseudodata_generation.csv', dat, delimiter = ',', fmt='%.7f')
#print(dat)
np.savetxt('pseudodata_generation.csv', dat, delimiter = ',', fmt='%.7f')
