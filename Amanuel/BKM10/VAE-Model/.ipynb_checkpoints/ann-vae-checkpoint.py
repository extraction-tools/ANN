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

import matplotlib
import matplotlib.pyplot as plt
from vae import Encoder, Decoder, VAENet

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

dats = pd.read_csv('pseudo_KM15.csv')
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
min_t = -1.*t
phi = np.array(dats['phi_x'])
F = np.array(dats['F'])
errF_temp = np.array(dats['sigmaF'])
errF = 1.0 * errF_temp

#Use the following two lines if you use real data. Comment out if you use pseudo data
#F1 = np.array(dats['F1'])
#F2 = np.array(dats['F2'])

#Use following line if you use pseudo data. Comment out if you use real data
F1_term = tb.Get_F1
F2_term = tb.Get_F2
F1 = F1_term(t)
F2 = F2_term(t)

# The true CFFs
ReH = np.array(dats['ReH'])
ReE = np.array(dats['ReE'])
ReHTilde = np.array(dats['ReHTilde'])
dvcs = np.array(dats['dvcs'])

#We don't use this for now:
def cosinus(x):
  return np.cos(x*3.1415926535/180. )

cosphi = cosinus(phi)

#Normalize variable. We normalize input into [-1,1]
k_norm = -1. + 2 * (k - 4.45) / (11.00 - 4.45)
qq_norm = -1. + 2 * (qq - 1.1) / (9. - 1.1)
xb_norm = -1. + 2 * (xb - 0.11) / (0.65 - 0.11)
t_norm = -1. + 2 * (t - (-1.4)) / (-0.1 - (-1.4))

#This is for plotting
df = pd.read_csv("pseudo_KM15.csv")

#Plotting F fit
def F2VsPhi(dataframe,SetNum,xdat,cffs):
    TempFvalSilces=dataframe[dataframe["#Set"]==SetNum]
    TempFvals=TempFvalSilces["F"]
    TempFvals_sigma=TempFvalSilces["sigmaF"]
    mask = (TempFvals_sigma[0:24] > 0)
    temp_phi=TempFvalSilces["phi_x"]
    plt.errorbar(temp_phi[mask],TempFvals[mask],TempFvals_sigma[mask],fmt='.',color='blue',label="Data")
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
    plt.title("Local fit with data set #"+str(SetNum)+"\nModel: Baseline Network," \
              + " RMSE: " + rmse + ", MAE: " + mae ,fontsize=12)
    plt.plot(temp_phi[mask], f(xdat,cffs), 'g--', label='fit')
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

numberOfSets = 100

hidden_dim = 400
latent_dim = 200

encoder = Encoder(input_dim=4, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim =4)


for ii in range(numberOfSets): # set how many sets to process
 datset = ii
 yrep = []
 
 #current architecture that I use
 blank_net = VAENet(Encoder=encoder, Decoder=decoder)

 optimizer = torch.optim.Adam(blank_net.parameters(), lr=0.0009)
 decayRate = 0.96
 my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

 EPOCH = 25000 #maximum epoch
  
 i = datset
 a = 24*i # start index of set
 b = a+24 # end index of set


 net = blank_net # untrain/reset network

 yrep = [0] * (b-a) # create array to be filled with replicated F values

 for l in range(b-a): # populate yrep with random normal values with mean = F and sd = errF

     yind = a+l # index of data point 
     #yrep[l] = (np.random.normal(F[yind], errF[yind])) #use this if you want to smear
     yrep[l] = F[yind] #use this instead of the line above if you dont want to smear the replica
 
 
 mask = (errF[a:b] > 0) #For masking. There are several points i csv file with error = 0 and F = 0
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

 # to track the loss as the model trains. We don't use it for now. Maybe useful for later
 ##train_losses = []
 ##valid_losses = []
 ##avg_train_losses = []
 ##avg_valid_losses = []
 ##losses = []
 ##losses.clear() 

 early_stopping = EarlyStopping(patience=25, verbose=False, delta = 0.0000005)
 for epoch in range(EPOCH):
    
     
     p, mean, log_var = net(x_norm.float())

     hs = torch.transpose(p, 0, 1)[0] 
     es = torch.transpose(p, 0, 1)[1] # array of 45 values for ReE at each increment of phi
     hts = torch.transpose(p, 0, 1)[2] 
     c0s = torch.transpose(p, 0, 1)[3]
     
     ReHfit = torch.mean(hs)
     ReEfit = torch.mean(es)
     ReHTfit = torch.mean(hts)
     c0fit = torch.mean(c0s)
    
     cffs = [ReHfit, ReEfit, ReHTfit, c0fit]
     
     loss = loss_func((xdat_var.float()), cffs, errs_var, y, mean, log_var)
     loss_val = loss_validation((xdat_var.float()), cffs, errs_var, y, mean, log_var)
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
 #F2VsPhi(df,ii+1,xdat,fit_cffs)
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
fig, axs = plt.subplots(2, 2, figsize=(8,8))
axs = axs.flatten()
x = []
for i in range(numberOfSets):
    x.append(i+1)

ReH_MAE = 0
ReE_MAE = 0
ReHTilde_MAE = 0
dvcs_MAE = 0

ReH_RMSE = 0
ReE_RMSE = 0
ReHTilde_RMSE = 0
dvcs_RMSE = 0

for i in range(numberOfSets//10):
    ReH_MAE += abs(ReH_all[i] - ReH[:numberOfSets][i])
    ReE_MAE += abs(ReE_all[i] - ReE[:numberOfSets][i])
    ReHTilde_MAE += abs(ReHT_all[i] - ReHTilde[:numberOfSets][i])
    dvcs_MAE += abs(c0fit_all[i] - dvcs[:numberOfSets][i])
    
    ReH_RMSE += (ReH_all[i] - ReH[:numberOfSets][i])**2
    ReE_RMSE += (ReE_all[i] - ReE[:numberOfSets][i])**2
    ReHTilde_RMSE += (ReHT_all[i] - ReHTilde[:numberOfSets][i])**2
    dvcs_RMSE += (c0fit_all[i] - dvcs[:numberOfSets][i])**2

ReH_MAE /= numberOfSets
ReE_MAE /= numberOfSets
ReHTilde_MAE /= numberOfSets
dvcs_MAE /= numberOfSets

ReH_RMSE =  np.sqrt(ReH_RMSE/numberOfSets)
ReE_RMSE =  np.sqrt(ReE_RMSE/numberOfSets)
ReHTilde_RMSE = np.sqrt(ReHTilde_RMSE/numberOfSets)
dvcs_RMSE =  np.sqrt(dvcs_RMSE/numberOfSets)

axs[0].scatter(x, ReH_all, label="Predicted")
axs[0].scatter(x, ReH[:numberOfSets], label="Data")
axs[0].set_ylabel("Re($\mathcal{H}$)", fontsize=14)
axs[0].set_xlabel("Set number", fontsize=12)
axs[0].set_title("Model: VAE\n MAE: " + "{:.4f}".format(ReH_MAE) + " RMSE: " + "{:.4f}".format(ReH_RMSE), fontsize=12)
axs[0].legend()

axs[1].scatter(x, ReE_all, label="Predicted")
axs[1].scatter(x, ReE[:numberOfSets], label="Data")
axs[1].set_ylabel("Re($\mathcal{E}$)", fontsize=14)
axs[1].set_xlabel("Set number", fontsize=14)
axs[1].set_title("Model: VAE\n MAE: " + "{:.4f}".format(ReE_MAE)+ " RMSE: " + "{:.4f}".format(ReE_RMSE), fontsize=12)
axs[1].legend()

axs[2].scatter(x, ReHT_all, label="Predicted")
axs[2].scatter(x, ReHTilde[:numberOfSets], label="Data")
axs[2].set_ylabel("Re($\mathcal{H}t$)", fontsize=14)
axs[2].set_xlabel("Set number", fontsize=14)
axs[2].set_title("Model: VAE\n MAE: " + "{:.4f}".format(ReHTilde_MAE)+ " RMSE: " + "{:.4f}".format(ReHTilde_RMSE), fontsize=12)
axs[2].legend()

axs[3].scatter(x, c0fit_all, label="Predicted")
axs[3].scatter(x, dvcs[:numberOfSets], label="Data")
axs[3].set_ylabel("$c_0$", fontsize=14)
axs[3].set_xlabel("Set number", fontsize=14)
axs[3].set_title("Model: VAE\n MAE: " + "{:.4f}".format(dvcs_MAE)+ " RMSE: " + "{:.4f}".format(dvcs_RMSE), fontsize=12)
axs[3].legend()

fig.tight_layout()

file_name = "VAE-CFFs-" + str(hidden_dim) + "-" + str(latent_dim) +".png"
plt.savefig(file_name)

dat = dat.T
np.savetxt('ResultData_HallA_all_local.txt', dat, delimiter = '\t', fmt='%.5f')
