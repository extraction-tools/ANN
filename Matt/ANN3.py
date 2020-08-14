import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

import numpy as np
import pandas as pd 
from BHDVCS_torch import TBHDVCS

#import BHDVCS_fit as dvcsfit

import matplotlib
import matplotlib.pyplot as plt

from scipy.stats import chisquare

tb = TBHDVCS()

f = tb.TotalUUXS_curve_fit3
loss_func = tb.loss_MSE

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
err_H = []
err_E = []
err_HT = []

EPOCH = 2500

for datset in range(14):
    a = datset*36
    b = a + 36


    xdat = np.asarray([phi[a:b], qq[a:b], xb[a:b], t[a:b], k[a:b], F1[a:b], F2[a:b], const[a:b]])
    x = Variable(torch.from_numpy(xdat[1:5].transpose())) # using qq, xb, t, k
    y = Variable(torch.from_numpy(ydat[a:b].transpose()))
    xdat = Variable(torch.from_numpy(xdat))
    errs = Variable(torch.from_numpy(errF[a:b]))

    net = torch.nn.Sequential(
        torch.nn.Linear(4, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 100),
        torch.nn.Tanh(),
        torch.nn.Linear(100, 80),
        torch.nn.Tanh(),
        torch.nn.Linear(80, 3)
    )

    optimizer = torch.optim.Adam(net.parameters(), lr=0.02)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    losses = []
    losses.clear()

    for epoch in range(EPOCH):

        p = net(x.float()) #output 3 predicted values for cffs

        ReHfit = torch.mean(torch.transpose(p, 0, 1)[0])
        ReEfit = torch.mean(torch.transpose(p, 0, 1)[1])
        ReHTfit = torch.mean(torch.transpose(p, 0, 1)[2])
        cffs = [ReHfit, ReEfit, ReHTfit]

        loss = loss_func((xdat.float()), cffs, errs, y)
        losses.append(float(loss.data.float()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plt.plot(np.linspace(int(.05*EPOCH), EPOCH, int(.95*EPOCH)), np.asarray(losses)[int(.05*EPOCH):], 'bo', label='Loss')
    # plt.plot(np.linspace(int(.05*EPOCH), EPOCH, int(.95*EPOCH)), np.zeros(int(0.95*EPOCH))+float(loss.data.float()), 'g--',             label='Final Loss = %.3e' % (float(loss.data.float())))
    # plt.legend()
    # plt.show()

    ReHfit = torch.mean(torch.transpose(p, 0, 1)[0]).data.numpy()
    ReEfit = torch.mean(torch.transpose(p, 0, 1)[1]).data.numpy()
    ReHTfit = torch.mean(torch.transpose(p, 0, 1)[2]).data.numpy()
    fit_cffs = [ReHfit, ReEfit, ReHTfit]

    # plt.plot(phi[a:b], ydat[a:b], 'bo', label='data')
    # plt.plot(phi[a:b], f(xdat,fit_cffs), 'g--', label='fit')
    # plt.legend()
    # plt.show()

    err_H.append(abs(100*(abs(fit_cffs[0]-ReH_target[a]))/ReH_target[a]))
    err_E.append(abs(100*(abs(fit_cffs[1]-ReE_target[a]))/ReE_target[a]))
    err_HT.append(abs(100*(abs(fit_cffs[2]-ReHT_target[a]))/ReHT_target[a]))

    print('Chi-Squared Value for this fit: %.3e' % (chisquare(f(xdat,fit_cffs), ydat[a:b])[0]))
    print('MSE Loss Value for this fit: %.3e' % (float(loss.data.float())))
    print('Average Error for set #%d using ANN = %.2f%%' % ((datset), ((err_H[-1]+err_E[-1]+err_HT[-1])/3)))
    #dvcsfit.fit_scipy(datset)

print('\n\033[1m%s%.2f%%' % ('Avg. Error of ReH = ', sum(err_H)/len(err_H)))
print('\033[1m%s%.2f%%' % ('Avg. Error of ReE = ', sum(err_E)/len(err_E)))
print('\033[1m%s%.2f%%' % ('Avg. Error of ReHT = ', sum(err_HT)/len(err_HT)))