import torch
import torch.nn as nn 
from torch.autograd import Variable 
import torch.nn.functional as F 
import torch.utils.data as Data

import numpy as np
import pandas as pd 
from BHDVCS_torch import TBHDVCS
from pytorchtools import EarlyStopping

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

tb = TBHDVCS()
f = tb.curve_fit

loss_func = tb.loss_MSE
loss_validation = tb.loss_chisq
loss_validation2 = tb.loss_MAE
loss_validation3 = tb.loss_MSE

dats = pd.read_csv('BKM_pseudodata.csv')

k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
min_t = -1.*t
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF = np.array(dats['sigmaF'])
F1 = np.array(dats['F1']) 
F2 = np.array(dats['F2'])
const = np.array(dats['dvcs'])

def cosinus(x):
  return np.cos(x*3.1415926535/180. )

cosphi = cosinus(phi)

df = pd.read_csv("BKM_pseudodata.csv")
def F2VsPhi(dataframe,SetNum,xdat,cffs):
    TempFvalSilces=dataframe[dataframe["#Set"]==SetNum]
    TempFvals=TempFvalSilces["F"]
    TempFvals_sigma=TempFvalSilces["sigmaF"]
    temp_phi=TempFvalSilces["phi_x"]
    plt.errorbar(temp_phi,TempFvals,TempFvals_sigma,fmt='.', color='blue', label="Data")
    plt.xlim(0,368)
    rmse = 0
    mae = 0
    pred = f(xdat,cffs).detach().numpy()
    actual = TempFvals.to_numpy()
    for i in range(len(actual)):
        rmse += (actual[i] - pred[i])**2
        mae += abs(actual[i] - pred[i])
    rmse = np.sqrt(( rmse / len(actual) ))
    mae /= len(actual)
    rmse = "{:.5f}".format(rmse)
    mae = "{:.5f}".format(mae)
    temp_unit=(np.max(TempFvals)-np.min(TempFvals))/len(TempFvals)
    plt.ylim(np.min(TempFvals)-temp_unit,np.max(TempFvals)+temp_unit)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc=4,fontsize=10,handlelength=3)
    plt.xlabel('$\phi_x$')
    plt.ylabel("F")
    plt.title("Local fit with data set #"+str(SetNum)+"\nModel: Baseline Network," \
              + " RMSE: " + rmse + ", MAE: " + mae ,fontsize=12)
    plt.plot(temp_phi, pred, 'r--', label="Fit")
    plt.legend(loc="upper center")
    file_name = "plots/plot_set_number_{}.png".format(SetNum)
    plt.savefig(file_name)

by_set = []

for ii in range(15): # set how many sets to process
 datset = ii
 yrep = []


 #current architecture that I use
 blank_net = torch.nn.Sequential(
         torch.nn.Linear(4, 100),
         #torch.nn.Tanh(),
         #torch.nn.Dropout(0.25),
         #torch.nn.Linear(200, 200), 
         #torch.nn.Tanhshrink(),
         torch.nn.Linear(100, 100),
         ##torch.nn.Tanhshrink(),
         torch.nn.Tanh(),
         #torch.nn.Dropout(0.35),
         torch.nn.Linear(100, 100),
         torch.nn.Tanh(),
         torch.nn.Linear(100, 100),
         torch.nn.Tanh(),
         torch.nn.Linear(100, 4)
 )


 optimizer = torch.optim.Adam(blank_net.parameters(), lr=0.0085)
 decayRate = 0.96
 my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

 EPOCH = 15000 #maximum epoch

 fVals = np.zeros((100,45))
 fitErrs = np.zeros((45))

 i = datset
 a = 45*i # start index of set
 b = a+45 # end index of set


 net = blank_net # untrain/reset network

 yrep = [0] * (b-a) # create array to be filled with replicated F values

 for l in range(b-a): # populate yrep with random normal values with mean = F and sd = errF

     yind = a+l # index of data point 
     yrep[l] = (np.random.normal(F[yind], errF[yind]))
     #yrep[l] = F[yind] #use this instead of the line above if you dont want to smear the replica
 
 xdat = np.array([phi[a:b], qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b]])
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
 losses = []
 losses.clear() 
 early_stopping = EarlyStopping(patience=25, verbose=False, delta = 0.0000005)
 for epoch in range(EPOCH):

     p = net(x.float()) #output arrays for 3 predicted values for cffs

     hs = torch.transpose(p, 0, 1)[0] 
     es = torch.transpose(p, 0, 1)[1] # array of 45 values for ReE at each increment of phi
     hts = torch.transpose(p, 0, 1)[2] 
     c1s = torch.transpose(p, 0, 1)[3]
     
     ReHfit = torch.mean(hs)
     ReEfit = torch.mean(es)
     ReHTfit = torch.mean(hts)
     c1fit = torch.mean(c1s)
    
     

     cffs = [ReHfit, ReEfit, ReHTfit, c1fit]

     loss = loss_func((xdat.float()), cffs, errs, y)
     loss_val = loss_validation((xdat.float()), cffs, errs, y)
     loss_val2 = loss_validation2((xdat.float()), cffs, errs, y)
     loss_val3 = loss_validation3((xdat.float()), cffs, errs, y)
     losses.append(float(loss.data.float()))
     
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     #my_lr_scheduler.step() #uncomment this line if you use learning scheduller


     #valid_losses.append(loss.item())
     
     # uncomment these following 3 lines below if you want to apply early stopping
     ##early_stopping(loss, net)
     ##if early_stopping.early_stop:
     ##     break

 ReHfits = (torch.transpose(p, 0, 1)[0]).data.numpy()
 ReEfits = (torch.transpose(p, 0, 1)[1]).data.numpy()
 ReHTfits = (torch.transpose(p, 0, 1)[2]).data.numpy()
 c1fits = (torch.transpose(p, 0, 1)[3]).data.numpy()
 

 ReHfit = np.mean(ReHfits)
 ReEfit = np.mean(ReEfits)
 ReHTfit = np.mean(ReHTfits)
 c1fit = np.mean(c1fits)
 
 
 fit_cffs = [ReHfit, ReEfit, ReHTfit,c1fit]

 by_set.append(fit_cffs)

 print('%.4f %.4f %.4f %.8f %.4f %.4f %.8f' % (ReHfit, ReEfit, ReHTfit, loss, loss_val, loss_val2, loss_val3))
 F2VsPhi(df,ii+1,xdat,fit_cffs)
 plt.clf()
 
df = pd.DataFrame(by_set)
df.to_csv('bySetCFFs.csv')

 ##these few lines below are to make loss vs epoch plot. The plot is ugly, need improvement
 #plt.plot(np.linspace(int(.05*EPOCH), EPOCH, int(.95*EPOCH)), np.asarray(losses)[int(.05*EPOCH):], 'bo', label='Loss')
 #plt.plot(np.linspace(int(.05*EPOCH), EPOCH, int(.95*EPOCH)), np.zeros(int(0.95*EPOCH))+float(loss.data.float()), 'g--',             label='Final Loss = %.3e' % (float(loss.data.float())))
 #plt.legend()
 #plt.show()
 #file_name = "loss_plot_set_{}.png".format(ii)
 #plt.savefig(file_name)  
