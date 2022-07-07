import torch
import torch.nn as nn 
from torch.autograd import Variable 
import torch.nn.functional as F 
import torch.utils.data as Data

import numpy as np
import pandas as pd 
from BHDVCS_torch import TBHDVCS
from BHDVCS_fit import DVCSFIT
from pytorchtools import EarlyStopping
from rbf import RBFLayer, RBFNet, RBFNet2Layer
from sklearn.cluster import KMeans

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare


tb = TBHDVCS()

dv = DVCSFIT()
func_fit = dv.curve_fit2
f = dv.plot_fit
BHC0 = dv.BHplusC0

loss_func = tb.loss_chisq
loss_validation = tb.loss_chisq
loss_validation2 = tb.loss_MAE
loss_validation3 = tb.loss_MSE

dats = pd.read_csv('data_hallA_allv2.csv')
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
min_t = -1.*t
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF_temp = np.array(dats['sigmaF'])
errF = 1.0 * errF_temp
F1 = np.array(dats['F1'])
F2 = np.array(dats['F2'])

#We don't use this for now:
def cosinus(x):
  return np.cos(x*3.1415926535/180. )

cosphi = cosinus(phi)

#Normalize variable. We normalize input into [-1,1]
k_norm = -1. + 2 * (k - 4.45) / (11.00 - 4.45)
qq_norm = -1. + 2 * (qq - 1.1) / (9. - 1.1)
xb_norm = -1. + 2 * (xb - 0.11) / (0.65 - 0.11)
t_norm = -1. + 2 * (t - (-1.4)) / (-0.1 - (-1.4))

mask = (errF > 0)
kmeansInput = np.array([qq_norm[mask], xb_norm[mask], t_norm[mask], k_norm[mask] ])
# create the centers used in the activation functions of each neuron
kmeans = KMeans(n_clusters=60, random_state=0).fit(kmeansInput.reshape(-1,1)) 
centers1 = kmeans.cluster_centers_
kmeans = KMeans(n_clusters=60, random_state=1).fit(kmeansInput.reshape(-1,1)) 
centers2 = kmeans.cluster_centers_

#This is for plotting
df = pd.read_csv("data_hallA_allv2.csv")

#Plotting F fit
def F2VsPhi(dataframe,SetNum,xdat,cffs):
    TempFvalSilces=dataframe[dataframe["#Set"]==SetNum]
    TempFvals=TempFvalSilces["F"]
    TempFvals_sigma=TempFvalSilces["sigmaF"]
    temp_phi=TempFvalSilces["phi_x"]
    plt.errorbar(temp_phi,TempFvals,TempFvals_sigma,fmt='.', color='blue', label="Data")
    plt.xlim(0,368)
    rmse = 0
    mae = 0
    pred = f(xdat,cffs)
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
    plt.title("Local fit with data set #"+str(SetNum)+"\nModel: RBF Network," \
              + " RMSE: " + rmse + ", MAE: " + mae ,fontsize=12)
    plt.plot(temp_phi, pred, 'r--', label="Fit")
    plt.legend(loc="upper center")
    file_name = "plots/plot_set_number_{}.png".format(SetNum)
    plt.savefig(file_name)

#Plotting interference term
def F2VsPhi_r(dataframe,SetNum,xdat,cffs): # F minus BH and C0 term
    TempFvalSilces=dataframe[dataframe["#Set"]==SetNum]
    TempFvals=TempFvalSilces["F"]
    TempFvals_sigma=TempFvalSilces["sigmaF"]
    mask = (TempFvals_sigma[0:24] > 0)
    temp_phi=TempFvalSilces["phi_x"]
    plt.errorbar(temp_phi[mask],TempFvals[mask] - BHC0(xdat, cffs),TempFvals_sigma[mask],fmt='.',color='blue',label="Data")
    plt.xlim(0,368)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc=4,fontsize=10,handlelength=3)
    plt.title("Interference term with data set #"+str(SetNum),fontsize=20)
    plt.plot(temp_phi[mask], (f(xdat, cffs) - BHC0(xdat, cffs)), 'g--', label='fit')
    file_name = "plot_set_number_{}.png".format(SetNum)
    plt.savefig(file_name)

#Various function. Mostly you can ignore it. This is where I experimented with various loss. Also for checking between pytorch and numpy based function
def LSQ(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.sum((ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ** 2 / errs ** 2)

def LSQ_exp(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.abs(1 - ((np.sum((ydat - 1000*func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ** 2 / errs ** 2)) / 20. ))

def MLM(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.sum(1000*func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit) - ydat + ydat * np.log(ydat/(1000*func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit))))

def MLM2(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.sum( -1. * ydat * np.log(func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)))

def MLM3(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.sum(func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)  - ydat * np.log(func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)))

def MLM4(ReH, ReE, ReHtilde, c0fit, c1fit):
    n_tot = 1000*np.sum(ydat)
    f_tot = 1000*np.sum(1000*func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit))
    return -n_tot * np.log(f_tot)  + f_tot -  np.sum(1000 * ydat * np.log(1000*func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit) / np.sum(1000*func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ))

def chisq(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.mean((ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ** 2 / errs ** 2)

def chisq(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.mean((ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ** 2 / errs ** 2)

def MAE(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.mean(np.abs(ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit))/ydat)

def MSE(ReH, ReE, ReHtilde, c0fit, c1fit):
    return np.mean((ydat - func_fit(xdat, ReH, ReE, ReHtilde, c0fit, c1fit)) ** 2 )


ReH_all = np.array([])
ReE_all = np.array([])
ReHT_all = np.array([])
c0fit_all = np.array([])

for ii in range(10): # set how many sets to process
 datset = ii
 yrep = []
 


 EPOCH = 25000 #maximum epoch
  
 i = datset
 a = 24*i # start index of set
 b = a+24 # end index of set

 

 yrep = [0] * (b-a) # create array to be filled with replicated F values

 for l in range(b-a): # populate yrep with random normal values with mean = F and sd = errF

     yind = a+l # index of data point 
     #yrep[l] = (np.random.normal(F[yind], errF[yind])) #use this if you want to smear
     yrep[l] = F[yind] #use this instead of the line above if you dont want to smear the replica
 
 
 mask = (errF[a:b] > 0) #For masking. There are several points in the csv file with error = 0 and F = 0
 xdat = np.array([phi[a:b][mask], qq[a:b][mask], xb[a:b][mask], t[a:b][mask], k[a:b][mask], F1[a:b][mask], F2[a:b][mask]])
 ydat = np.array(yrep)[mask]
 errs = np.array([errF[a:b][mask]])
 
 x = Variable(torch.from_numpy(xdat[1:5].transpose()))
 y = Variable(torch.from_numpy(ydat.transpose()))

 xdat_var = Variable(torch.from_numpy(xdat))
 errs_var = Variable(torch.from_numpy(errF[a:b][mask]))

 #Normalize variable
 xdat_norm = np.array([phi[a:b][mask], qq_norm[a:b][mask], xb_norm[a:b][mask], t_norm[a:b][mask], k_norm[a:b][mask], F1[a:b][mask], F2[a:b][mask]])
 
 x_norm = Variable(torch.from_numpy(xdat_norm[1:5].transpose()))
 
    
 net = RBFNet2Layer(4, 4, centers1, centers2, True) # untrain/reset network

 optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
 decayRate = 0.96
 my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
    
 # to track the loss as the model trains. We don't use it for now. Maybe useful for later
 ##train_losses = []
 ##valid_losses = []
 ##avg_train_losses = []
 ##avg_valid_losses = []
 ##losses = []
 ##losses.clear() 

 early_stopping = EarlyStopping(patience=25, verbose=False, delta = 0.0000005)
 for epoch in range(EPOCH):

     
     p = net(x_norm.float())
     
     hs = torch.transpose(p, 0, 1)[0] 
     es = torch.transpose(p, 0, 1)[1] # array of 45 values for ReE at each increment of phi
     hts = torch.transpose(p, 0, 1)[2] 
     c0s = torch.transpose(p, 0, 1)[3]
     
     ReHfit = torch.mean(hs)
     ReEfit = torch.mean(es)
     ReHTfit = torch.mean(hts)
     c0fit = torch.mean(c0s)
    
     cffs = [ReHfit, ReEfit, ReHTfit, c0fit]

     loss = loss_func((xdat_var.float()), cffs, errs_var, y)
     loss_val = loss_validation((xdat_var.float()), cffs, errs_var, y)
     loss_val2 = loss_validation2((xdat_var.float()), cffs, errs_var, y)
     loss_val3 = loss_validation3((xdat_var.float()), cffs, errs_var, y)
     ##losses.append(float(loss.data.float())) # not using for now but maybe useful for later
     
     #print('%.4f %.8f %.4f %.4f %.8f' % (epoch,  loss, loss_val, loss_val2, loss_val3))
     optimizer.zero_grad()
     loss.backward()
     optimizer.step()
     ##my_lr_scheduler.step() #uncomment this line if you want use learning scheduller


     #valid_losses.append(loss.item()) # maybe useful for later
     
     # uncomment these following 3 lines below if you want to apply early stopping
     #early_stopping(loss, net)
     #if early_stopping.early_stop:
          #break
     #Early stopping that currently I applied
     if loss < 1.01 and loss > 0.99:
       break


 ReHfits = (torch.transpose(p, 0, 1)[0]).data.numpy()
 ReEfits = (torch.transpose(p, 0, 1)[1]).data.numpy()
 ReHTfits = (torch.transpose(p, 0, 1)[2]).data.numpy()
 c0fits = (torch.transpose(p, 0, 1)[3]).data.numpy()
 

 ReHfit = np.mean(ReHfits)
 ReEfit = np.mean(ReEfits)
 ReHTfit = np.mean(ReHTfits)
 c0fit = np.mean(c0fits)
 c1fit = 0.
 
 fit_cffs = [ReHfit, ReEfit, ReHTfit,c0fit, c1fit]
 ReH_all = np.append(ReH_all, ReHfit)
 ReE_all = np.append(ReE_all, ReEfit)
 ReHT_all = np.append(ReHT_all, ReHTfit)
 c0fit_all = np.append(c0fit_all, c0fit)
 print('%.4f %.4f %.4f %.4f %.8f %.4f %.4f %.8f' % (ReHfit, ReEfit, ReHTfit, c0fit, loss, loss_val, loss_val2, loss_val3))

 ## loss-calculation cross check  using numpy to compare with torch
 loss_val_np = chisq(ReHfit, ReEfit, ReHTfit, c0fit, c1fit)
 loss_val2_np = MAE(ReHfit, ReEfit, ReHTfit, c0fit, c1fit)
 loss_val3_np = MSE(ReHfit, ReEfit, ReHTfit, c0fit, c1fit)
 print('%.4f %.4f %.4f %.4f %.8f' % (ii, epoch, loss_val_np, loss_val2_np, loss_val3_np))
 F2VsPhi(df,ii+1,xdat,fit_cffs)
 plt.clf()

 ##these few lines below are to make loss vs epoch plot. The plot is still ugly, need improvement
 #plt.plot(np.linspace(int(.05*EPOCH), EPOCH, int(.95*EPOCH)), np.asarray(losses)[int(.05*EPOCH):], 'bo', label='Loss')
 #plt.plot(np.linspace(int(.05*EPOCH), EPOCH, int(.95*EPOCH)), np.zeros(int(0.95*EPOCH))+float(loss.data.float()), 'g--', label='Final Loss = %.3e' % (float(loss.data.float())))
 #plt.legend()
 #plt.show()
 #file_name = "loss_plot_set_{}.png".format(ii)
 #plt.savefig(file_name)  

## save in txt
dat = np.array([ReH_all, ReE_all, ReHT_all, c0fit_all])
dat = dat.T
np.savetxt('ResultData_HallA_all_local.txt', dat, delimiter = '\t', fmt='%.5f')
