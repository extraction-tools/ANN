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
from scipy import stats
from random import seed
from random import random
import math

tb = TBHDVCS()
loss_global = tb.loss_chisq_mean
loss_func = tb.loss_chisq_mean
F1_term = tb.Get_F1
F2_term = tb.Get_F2
GetF = tb.getF

FileName = "pseudo_KM15.csv"
dats = pd.read_csv(FileName)
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
min_t = -1.*t
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF_temp = np.array(dats['sigmaF'])
errF = 1.0 * errF_temp
gReH = np.array(dats['ReH'])
gReE = np.array(dats['ReE'])
gReHTilde = np.array(dats['ReHTilde'])
gc0fit = np.array(dats['dvcs'])

F1 = F1_term(t)
F2 = F2_term(t)

##to get start index
dataframe = pd.read_csv(FileName)
N_data = []
start_index = []
n_total = 0
for i in range(195):
  TempFvalSilces=dataframe[dataframe["#Set"]==i+1]
  TempFvals=TempFvalSilces["F"]
  start_index = np.append(start_index, n_total)
  N_data = np.append(N_data, TempFvals.size)
  n_total = n_total + TempFvals.size


#Normalize input
k_norm = -1. + 2 * (k - 4.45) / (11.00 - 4.45)
qq_norm = -1. + 2 * (qq - 1.1) / (9. - 1.1)
xb_norm = -1. + 2 * (xb - 0.11) / (0.65 - 0.11)
t_norm = -1. + 2 * (t - (-1.4)) / (-0.1 - (-1.4))

#Various F_replica option
F_replica = np.random.normal(F, errF)
#F_replica = F #No smearing

#split the set for training anf validation. 1 is training 0 is validation set (optional)
train_index = np.array([])
#seed(1)
for ii in range(0,195):
  rand_value = random()
  if rand_value > 0.15:
    train_index = np.append(train_index, 1)
  else:
    train_index = np.append(train_index, 0)


#here to play the neuron architecture
blank_net = torch.nn.Sequential(
         torch.nn.Linear(3, 100),
         torch.nn.Linear(100, 100),
         torch.nn.Tanhshrink(),
         torch.nn.Linear(100, 100),
         torch.nn.Tanhshrink(),
         torch.nn.Linear(100, 100),
         torch.nn.Tanh(), #usually Tanh
         torch.nn.Linear(100, 4)
     )

#option for weight initialization
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight,-0.25,0.25)

#optimizer & learning rate/scheduler
optimizer = torch.optim.Adam(blank_net.parameters(), lr=0.0005) #0.002
decayRate = 0.5

#learning rate scheduler
#my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5000,10000, 15000,20000,22500,25000, 50000,90000], gamma=decayRate) #long epoch
my_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100,200,300,400,500,600,800,1250,2000,4500, 6000, 8000], gamma=decayRate)


net = blank_net # untrain/reset network


#early stopping method 1
#early_stopping = EarlyStopping(patience=25, verbose=False, delta = 0.0005)

# Early stopping method 2
the_last_loss = 5000000
patience = 5
trigger_times = 0

arr_val = np.array([])
arr_epoch = np.array([])

import random

#Fixed or dynamics epoch for each replica
#EPOCH = random.randrange(27500, 227500, 25000)
#EPOCH = int(EPOCH)
EPOCH = 52500

mask = (F > 0)
ydat = np.array(F_replica)[mask]
y = Variable(torch.from_numpy(ydat.transpose()))

#Normalize variable for 3 input variables (no k)
xdat_norm_3 = np.array([qq_norm[mask], xb_norm[mask], t_norm[mask]])
x_norm_3 = Variable(torch.from_numpy(xdat_norm_3.transpose()))
xdat = np.array([phi[mask], qq[mask], xb[mask], t[mask], k[mask], F1[mask], F2[mask]])
xdat_var = Variable(torch.from_numpy(xdat))
errs_var = Variable(torch.from_numpy(errF[mask]))

for epoch in range(EPOCH):

  p = net(x_norm_3.float()) #output arrays for 4 predicted values for cffs
  hs = torch.transpose(p, 0, 1)[0]
  es = torch.transpose(p, 0, 1)[1] # array of 45 values for ReE at each increment of phi
  hts = torch.transpose(p, 0, 1)[2]
  c0s = torch.transpose(p, 0, 1)[3]

  ReHfit = torch.mean(hs)
  ReEfit = torch.mean(es)
  ReHTfit = torch.mean(hts)
  c0fit = torch.mean(c0s)
 
  cffs = [hs, es, hts, c0s]
  loss = loss_global((xdat_var.float()), cffs, errs_var, y)
  
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  #activate or deactivate learning scheduler
  #my_lr_scheduler.step()

  print('%.4f %.4f %i' %(epoch, loss, EPOCH))
  
ReH_all = np.array([])
ReE_all = np.array([])
ReHT_all = np.array([])
c0fit_all = np.array([])
epoch_all = np.array([])

#Getting results for 195 data sets
for ii in range(0, 195): # set how many sets to process
  datset = ii
  i = datset
  a = int(start_index[i]) # start index of set
  b = a + int( N_data[i]) # end index of set
 
  mask = (F[a:b] > 0)
  xdat_norm_3 = np.array([qq_norm[a:b][mask], xb_norm[a:b][mask], t_norm[a:b][mask]])
  x_norm_3 = Variable(torch.from_numpy(xdat_norm_3.transpose()))
  p = net(x_norm_3.float()) #output arrays for 4 predicted values for cffs
  ReHfits = (torch.transpose(p, 0, 1)[0]).data.numpy()
  ReEfits = (torch.transpose(p, 0, 1)[1]).data.numpy()
  ReHTfits = (torch.transpose(p, 0, 1)[2]).data.numpy()
  c0fits = (torch.transpose(p, 0, 1)[3]).data.numpy()
  ReHfit = np.mean(ReHfits)
  ReEfit = np.mean(ReEfits)
  ReHTfit = np.mean(ReHTfits)
  c0fit = np.mean(c0fits)

  ReH_all = np.append(ReH_all, ReHfit)
  ReE_all = np.append(ReE_all, ReEfit)
  ReHT_all = np.append(ReHT_all, ReHTfit)
  c0fit_all = np.append(c0fit_all, c0fit)
  epoch_all = np.append(epoch_all, EPOCH)
  
## save in txt
dat = np.array([ReH_all, ReE_all, ReHT_all, c0fit_all, epoch_all])
dat = dat.T
np.savetxt('ResultGlobalSet.txt', dat, delimiter = '\t', fmt='%.5f')

ReH_band = np.array([])
ReE_band = np.array([])
ReHT_band = np.array([])

#Getting results for 10000 points = 100 bin in xb * 100 bins in t or 2D bands
for x in range(100):
  for y in range(100):
   inp_qq = 8.0
   inp_xb = 0.15 + 0.005*x + 0.0025
   inp_t = -1.35 + 0.012*y + 0.006
   inp_qq = -1. + 2 * (inp_qq - 1.1) / (9. - 1.1)
   inp_xb = -1. + 2 * (inp_xb - 0.11) / (0.65 - 0.11)
   inp_t = -1. + 2 * (inp_t - (-1.4)) / (-0.1 - (-1.4))
   xdat_norm_3 = np.array([inp_qq, inp_xb, inp_t])
   x_norm_3 = Variable(torch.from_numpy(xdat_norm_3.transpose()))
   p = net(x_norm_3.float())
   ReH_band = np.append(ReH_band, p[0].detach().numpy())  
   ReE_band = np.append(ReE_band, p[1].detach().numpy())
   ReHT_band = np.append(ReHT_band, p[2].detach().numpy())

## save in txt part2
dat2 = np.array([ReH_band, ReE_band, ReHT_band])
dat2 = dat2.T
np.savetxt('ResultGlobalBand.txt', dat2, delimiter = '\t', fmt='%.5f')
