import torch
import torch.nn as nn 
from torch.autograd import Variable 
import torch.nn.functional as F 
import torch.utils.data as Data

import numpy as np
import pandas as pd 
from BHDVCS_torch_v2 import TBHDVCS
from pytorchtools import EarlyStopping

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

tb = TBHDVCS()

f = tb.TotalUUXS_curve_fit3
g = tb.TotalUUXS_curve_fit
loss_func = tb.loss_MSE

dats = pd.read_csv('dvcs_xs_May-2021_342_sets.csv')
n = np.array(dats['#Set'])
ind = np.array(dats['index'])
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF = np.array(dats['sigmaF'])
varF = np.array(dats['varF']) 
F1 = np.array(dats['F1'])
F2 = np.array(dats['F2'])
const = np.array(dats['dvcs'])  

def cosinus(x):
  return np.cos(x*3.1415926535/180. )

cosphi = cosinus(phi)

for ii in range(15):
 datset = ii
 yrep = []
 errs_H = []
 errs_E = []
 errs_HT = []


 blank_net = torch.nn.Sequential(
         torch.nn.Linear(4, 1024),
         torch.nn.Tanh(),
         #torch.nn.Dropout(0.15),
         torch.nn.Linear(1024, 104),
         torch.nn.Tanh(),
         #torch.nn.Dropout(0.15),
         #torch.nn.Linear(512, 104),
         #torch.nn.Tanh(),
         #torch.nn.Dropout(0.15),
         #torch.nn.Linear(102, 102),
         #torch.nn.Tanh(),
         #torch.nn.Dropout(0.15),
         torch.nn.Linear(104, 3)
     )

 optimizer = torch.optim.Adam(blank_net.parameters(), lr=0.008)
 decayRate = 0.96
 my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)


 EPOCH = 6000


 repNum=1


 fVals = np.zeros((100,45))
 fitErrs = np.zeros((45))

 i = datset
 a = 45*i # start index of set
 b = a+45 # end index of set

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
     #yrep[l] = F[yind]
 
 xdat = np.array([cosphi[a:b], qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b], const[a:b]])
 ydat = np.array(yrep)

 x = Variable(torch.from_numpy(xdat[1:5].transpose()))
 y = Variable(torch.from_numpy(ydat.transpose()))

 xdat = Variable(torch.from_numpy(xdat))
 errs = Variable(torch.from_numpy(errF[a:b]))

 # to track the loss as the model trains
 train_losses = []
 valid_losses = []
 avg_train_losses = []
 avg_valid_losses = [] 
 early_stopping = EarlyStopping(patience=20, verbose=False, delta = 0.000001)
 for epoch in range(EPOCH):

     p = net(x.float()) #output arrays for 3 predicted values for cffs

     hs = torch.transpose(p, 0, 1)[0] # array of 45 values for ReH at each increment of phi
     es = torch.transpose(p, 0, 1)[1] # array of 45 values for ReE at each increment of phi
     hts = torch.transpose(p, 0, 1)[2] # array of 45 values for ReHT at each increment of phi

     ReHfit = torch.mean(hs)
     ReEfit = torch.mean(es)
     ReHTfit = torch.mean(hts)

     cffs = [ReHfit, ReEfit, ReHTfit]

     loss = loss_func((xdat.float()), cffs, errs, y)

         
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     my_lr_scheduler.step()


     #valid_losses.append(loss.item())
     
     early_stopping(loss, net)
     if early_stopping.early_stop:
          break

 ReHfits = (torch.transpose(p, 0, 1)[0]).data.numpy()
 ReEfits = (torch.transpose(p, 0, 1)[1]).data.numpy()
 ReHTfits = (torch.transpose(p, 0, 1)[2]).data.numpy()
 ReHfit = np.mean(ReHfits)
 ReEfit = np.mean(ReEfits)
 ReHTfit = np.mean(ReHTfits)
 fit_cffs = [ReHfit, ReEfit, ReHTfit]

#     plt.plot(cosphi[a:b], F[a:b], 'ro', label='data')
#     plt.plot(cosphi[a:b], f(xdat,fit_cffs), 'b--', label='fit')
#     plt.legend()
#     plt.show()


 print('%.2f %.2f %.2f' % (ReHfits[18], ReEfits[18], ReHTfits[18]))

