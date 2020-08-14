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

from scipy.stats import chisquare

import time
import sys

tb = TBHDVCS()

f = tb.TotalUUXS_curve_fit3
loss_func = tb.loss_MSE

datset = int(sys.argv[1])

dats = pd.read_csv('dvcs_psuedo.csv')
n = np.array(dats['#Set'])
ind = np.array(dats['index'])
k = np.array(dats['k'])
qq = np.array(dats['QQ'])
xb = np.array(dats['x_b'])
t = np.array(dats['t'])
phi = np.array(dats['phi_x'])
ydat = np.array(dats['F'])
errF = np.array(dats['errF']) 
F1 = np.array(dats['F1'])
F2 = np.array(dats['F2'])
const = np.array(dats['dvcs'])  
ReH_target = np.array(dats['ReH']) 
ReE_target = np.array(dats['ReE']) 
ReHT_target = np.array(dats['ReHtilde'])

epoch_ops = [1500, 2500]
node_ops = [5, 50, 100]
learning_rate_ops = [0.005, 0.05, 0.5, 1]

def reset_nets(nodes):
    net2 = torch.nn.Sequential(
            torch.nn.Linear(4, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, 1)
        )


    net5 = torch.nn.Sequential(
            torch.nn.Linear(4, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, 1)
        )

    net10 = torch.nn.Sequential(
            torch.nn.Linear(4, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, nodes),
            torch.nn.Tanh(),
            torch.nn.Linear(nodes, 1)
        )

    return [net2, net5, net10]


best = 100

a = datset*36
b = a + 36


xdat = np.asarray([phi[a:b], qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b], const[a:b]])
x = Variable(torch.from_numpy(xdat[1:5].transpose()))
y = Variable(torch.from_numpy(ydat[a:b].transpose()))
xdat = Variable(torch.from_numpy(xdat))
errs = Variable(torch.from_numpy(errF[a:b]))



for lH in learning_rate_ops:
    for lE in learning_rate_ops:
        for lHT in learning_rate_ops:
            net_ops = reset_nets(30)
            layersReH=0
            for netReH in net_ops:
                layersReH+=1
                layersReE=0
                for netReE in net_ops:
                    layersReE+=1
                    layersReHT=0
                    for netReHt in net_ops:
                        layersReHT+=1
                        
                        optimizerH = torch.optim.Adam(netReH.parameters(), lr=lH)
                        optimizerE = torch.optim.Adam(netReE.parameters(), lr=lE)
                        optimizerHT = torch.optim.Adam(netReHt.parameters(), lr=lHT)

                        for epoch in range(2500):
                            predReH = netReH(x.float()) #output 3 predicted values for cffs
                            predReE = netReE(x.float())
                            predReHT = netReHt(x.float())

                            ReHfit = torch.mean(torch.transpose(predReH, 0, 1)[0])
                            ReEfit = torch.mean(torch.transpose(predReE, 0, 1)[0])
                            ReHTfit = torch.mean(torch.transpose(predReHT, 0, 1)[0])
                            cffs = [ReHfit, ReEfit, ReHTfit]

                            loss = loss_func((xdat.float()), cffs, errs, y)

                            optimizerH.zero_grad()
                            optimizerE.zero_grad()
                            optimizerHT.zero_grad()
                            loss.backward()
                            optimizerH.step()
                            optimizerE.step()
                            optimizerHT.step()

                        ReHfit = torch.mean(torch.transpose(predReH, 0, 1)[0]).data.numpy()
                        ReEfit = torch.mean(torch.transpose(predReE, 0, 1)[0]).data.numpy()
                        ReHTfit = torch.mean(torch.transpose(predReHT, 0, 1)[0]).data.numpy()
                        fit_cffs = [ReHfit, ReEfit, ReHTfit]
                        
                        err_H = abs(100*((fit_cffs[0]-ReH_target[a]))/ReH_target[a])
                        err_E = abs(100*((fit_cffs[1]-ReE_target[a]))/ReE_target[a])
                        err_HT = abs(100*((fit_cffs[2]-ReHT_target[a]))/ReHT_target[a])
                        
                        avgErr = (err_H + err_E + err_HT)/3
                        
                        if avgErr < best:
                            best = avgErr
                            optims = [layersReH, layersReE, layersReHT, lH, lE, lHT]
                            print('Optimized set #%d to %.2f%%' % (datset, best))
                            print('Optimizal architecture for set %d: %d layers ReH, %d layers ReH, %d layers ReH, %f lrH, %f lrE, %f lrHT.' % (datset, optims[0], optims[1], optims[2], optims[3], optims[4], optims[5]))
                            
