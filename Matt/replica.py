import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np
import pandas as pd 
from BHDVCS_torch import TBHDVCS

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

tb = TBHDVCS()

f = tb.TotalUUXS_curve_fit3
g = tb.TotalUUXS_curve_fit
loss_func = tb.loss_MSE_errs

dats = pd.read_csv('dvcs_psuedo.csv')
n = np.array(dats['#Set'])
ind = np.array(dats['index'])
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF = np.array(dats['errF']) 
F1 = np.array(dats['F1'])
F2 = np.array(dats['F2'])
const = np.array(dats['dvcs'])  
ReH_target = np.array(dats['ReH']) 
ReE_target = np.array(dats['ReE']) 
ReHT_target = np.array(dats['ReHtilde'])
yrep = []

errs_H = []
errs_E = []
errs_HT = []


blank_net = torch.nn.Sequential(
        torch.nn.Linear(4, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 80),
        torch.nn.Tanh(),
        torch.nn.Linear(80, 3)
    )

optimizer = torch.optim.Adam(blank_net.parameters(), lr=0.02)

EPOCH = 2500

#repNum = int(sys.argv[1])
repNum=1
datset = 0

fVals = np.zeros((100,36))
fitErrs = np.zeros((36))

i = datset
a = 36*i # start index of set
b = a+36 # end index of set

rep_ReH = []
rep_ReE = []
rep_ReHT = []
rep_ReH.clear()
rep_ReE.clear()
rep_ReHT.clear()




net = blank_net # untrain/reset network

yrep = [0] * (b-a) # create array to be filled with replicated F values

for l in range(b-a): # populate yrep with random normal values with mean = F and sd = errF

    yind = a+l # index of data point 
    yrep[l] = (np.random.normal(F[yind], errF[yind]))


xdat = np.array([phi[a:b], qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b], const[a:b]])
ydat = np.array(yrep)

x = Variable(torch.from_numpy(xdat[1:5].transpose()))
y = Variable(torch.from_numpy(ydat.transpose()))

xdat = Variable(torch.from_numpy(xdat))

errs = Variable(torch.from_numpy(errF[a:b]))

for epoch in range(EPOCH):

    p = net(x.float()) #output arrays for 3 predicted values for cffs

    hs = torch.transpose(p, 0, 1)[0] # array of 36 values for ReH at each increment of phi
    es = torch.transpose(p, 0, 1)[1] # array of 36 values for ReE at each increment of phi
    hts = torch.transpose(p, 0, 1)[2] # array of 36 values for ReHT at each increment of phi

    ReHfit = torch.mean(hs)
    ReEfit = torch.mean(es)
    ReHTfit = torch.mean(hts)

    cffs = [ReHfit, ReEfit, ReHTfit]

    loss = loss_func((xdat.float()), cffs, errs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


ReHfits = (torch.transpose(p, 0, 1)[0]).data.numpy()
ReEfits = (torch.transpose(p, 0, 1)[1]).data.numpy()
ReHTfits = (torch.transpose(p, 0, 1)[2]).data.numpy()
ReHfit = np.mean(ReHfits)
ReEfit = np.mean(ReEfits)
ReHTfit = np.mean(ReHTfits)
fit_cffs = [ReHfit, ReEfit, ReHTfit]

#     plt.plot(phi[a:b], F[a:b], 'ro', label='data')
#     plt.plot(phi[a:b], f(xdat,fit_cffs), 'b--', label='fit')
#     plt.legend()
#     plt.show()

print(ReHfits)
print('%d %.2f %.2f %.2f' % (repNum, ReHfits[18], ReEfits[18], ReHTfits[18]))

    


