import torch
import numpy as np

class TBHDVCS(object):

    def __init__(self):
        self.ALP_INV = 137.0359998 # 1 / Electromagnetic Fine Structure Constant
        self.PI = 3.1415926535
        self.RAD = self.PI / 180.
        self.M = 0.938272 #Mass of the proton in GeV
        self.GeV2nb = .389379*1000000 # Conversion from GeV to NanoBar
        self.M2 = self.M*self.M #Mass of the proton  squared in GeV
        

      
    def SetKinematics(self, _QQ, _x, _t, _k):

        self.QQ = _QQ #Q^2 value
        self.x = _x   #Bjorken x
        self.t = _t   #momentum transfer squared
        self.k = _k   #Electron Beam Energy
        
        self.ee = 4. * self.M2 * self.x * self.x / self.QQ # epsilon squared
        self.y = torch.sqrt(self.QQ) / ( torch.sqrt(self.ee) * self.k )  # lepton energy fraction
        self.xi = self.x * ( 1. + self.t / 2. / self.QQ ) / ( 2. - self.x + self.x * self.t / self.QQ ); # Generalized Bjorken variable
        self.Gamma1 = self.x * self.y * self.y / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / 8. / self.QQ / self.QQ / torch.sqrt( 1. + self.ee ) # factor in front of the cross section, eq. (22)
        self.s = 2. * self.M * self.k + self.M2
        self.tmin = - self.QQ * ( 2. * ( 1. - self.x ) * ( 1. - torch.sqrt(1. + self.ee) ) + self.ee ) / ( 4. * self.x * ( 1. - self.x ) + self.ee ) # eq. (31)
        self.K2 = - ( self.t / self.QQ ) * ( 1. - self.x ) * ( 1. - self.y - self.y * self.y * self.ee / 4.) * ( 1. - self.tmin / self.t ) * ( torch.sqrt(1. + self.ee) + ( ( 4. * self.x * ( 1. - self.x ) + self.ee ) / ( 4. * ( 1. - self.x ) ) ) * ( ( self.t - self.tmin ) / self.QQ )  ) # eq. (30)
        #___________________________________________________________________________________



    def BHLeptonPropagators(self, phi, _QQ, _x, _t, _k):
        self.SetKinematics(_QQ, _x, _t, _k)
	#K*D 4-vector product (phi-dependent)
        self.KD = - self.QQ / ( 2. * self.y * ( 1. + self.ee ) ) * ( 1. + 2. * torch.sqrt(self.K2) * torch.cos( self.PI - ( phi * self.RAD ) ) - self.t / self.QQ * ( 1. - self.x * ( 2. - self.y ) + self.y * self.ee / 2. ) + self.y * self.ee / 2.  ) #eq. (29)

  	# lepton BH propagators P1 and P2 (contaminating phi-dependence)
        self.P1 = 1. + 2. * self.KD / self.QQ
        self.P2 = self.t / self.QQ - 2. * self.KD / self.QQ
        
    

    def BHUU(self, phi, _QQ, _x, _t, _k, F1, F2) :

        self.BHLeptonPropagators(phi, _QQ, _x, _t, _k)
        #BH unpolarized Fourier harmonics eqs. (35 - 37)
        self.c0_BH = 8. * self.K2 * ( ( 2. + 3. * self.ee ) * ( self.QQ / self.t ) * ( F1 * F1  - F2 * F2 * self.t / ( 4. * self.M2 ) ) + 2. * self.x * self.x * ( F1 + F2 ) * ( F1 + F2 ) ) + ( 2. - self.y ) * ( 2. - self.y ) * ( ( 2. + self.ee ) * ( ( 4. * self.x * self.x * self.M2 / self.t ) * ( 1. + self.t / self.QQ ) * ( 1. + self.t / self.QQ ) + 4. * ( 1. - self.x ) * ( 1. + self.x * self.t / self.QQ ) ) * ( F1 * F1 - F2 * F2 * self.t / ( 4. * self.M2 ) ) + 4. * self.x * self.x * ( self.x + ( 1. - self.x + self.ee / 2. ) * ( 1. - self.t / self.QQ ) * ( 1. - self.t / self.QQ ) - self.x * ( 1. - 2. * self.x ) * self.t * self.t / ( self.QQ * self.QQ ) ) * ( F1 + F2 ) * ( F1 + F2 ) ) + 8. * ( 1. + self.ee ) * ( 1. - self.y - self.ee * self.y * self.y / 4. ) * ( 2. * self.ee * ( 1. - self.t / ( 4. * self.M2 ) ) * ( F1 * F1 - F2 * F2 * self.t / ( 4. * self.M2 ) ) - self.x * self.x * ( 1. - self.t / self.QQ ) * ( 1. - self.t / self.QQ ) * ( F1 + F2 ) * ( F1 + F2 ) )

        self.c1_BH = 8. * torch.sqrt(self.K2) * ( 2. - self.y ) * ( ( 4. * self.x * self.x * self.M2 / self.t - 2. * self.x - self.ee ) * ( F1 * F1 - F2 * F2 * self.t / ( 4. * self.M2 ) ) + 2. * self.x * self.x * ( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ ) * ( F1 + F2 ) * ( F1 + F2 ) )

        self.c2_BH = 8. * self.x * self.x * self.K2 * ( ( 4. * self.M2 / self.t ) * ( F1 * F1 - F2 * F2 * self.t / ( 4. * self.M2 ) ) + 2. * ( F1 + F2 ) * ( F1 + F2 ) )

  	#BH squared amplitude eq (25) divided by e^6
        self.Amp2_BH = 1. / ( self.x * self.x * self.y * self.y * ( 1. + self.ee ) * ( 1. + self.ee ) * self.t * self.P1 * self.P2 ) * ( self.c0_BH + self.c1_BH * torch.cos( self.PI - (phi * self.RAD) ) + self.c2_BH * torch.cos( 2. * ( self.PI - ( phi * self.RAD ) ) )  )

        self.Amp2_BH = self.GeV2nb * self.Amp2_BH #convertion to nb

  	#return self.dsigma_BH = self.Gamma1 * self.Amp2_BH
        return self.Gamma1 * self.Amp2_BH
        
        
    def IUU(self, phi, _QQ, _x, _t, _k, F1, F2, ReH, ReE, ReHtilde) :
        
        self.BHLeptonPropagators(phi, _QQ, _x, _t, _k)
        
        self.A = - 8. * self.K2 * ( 2. - self.y ) * ( 2. - self.y ) * ( 2. - self.y ) / ( 1. - self.y ) - 8. * ( 2. - self.y ) * ( 1. - self.y ) * ( 2. - self.x ) * self.t / self.QQ - 8. * torch.sqrt(self.K2) * ( 2. - 2. * self.y + self.y * self.y ) * torch.cos( self.PI - (phi * self.RAD) )
        self.B = 8. * self.x * self.x * ( 2. - self.y ) * (1 - self.y ) / ( 2. - self.x ) * self.t / self.QQ
        self.C =  self.x / ( 2. - self.x ) * ( - 8. * self.K2 * ( 2. - self.y ) * ( 2. - self.y ) * ( 2. - self.y ) / ( 1. - self.y ) - 8. * torch.sqrt(self.K2) * ( 2. - 2. * self.y + self.y * self.y ) * torch.cos( self.PI - (phi * self.RAD) ) )

  	# BH-DVCS interference squared amplitude eq (27) divided by e^6
        self.Amp2_I = 1. / ( self.x * self.y * self.y * self.y * self.t * self.P1 * self.P2 ) * ( self.A * ( F1 * ReH - self.t / 4. / self.M2 * F2 * ReE ) + self.B * ( F1 + F2 ) * ( ReH + ReE ) + self.C * ( F1 + F2 ) * ReHtilde )

        self.Amp2_I = self.GeV2nb * self.Amp2_I # convertion to nb

  	#return self.dsigma_I = self.Gamma1 * self.Amp2_I
        return self.Gamma1 * self.Amp2_I
    
    
    def loss_MSE(self, kins, cffs, errs, f_true):
        phi, kin1, kin2, kin3, kin4, F1, F2 = kins
        ReH, ReE, ReHtilde, c1fit = cffs #output of network

        self.SetKinematics(kin1, kin2, kin3, kin4)
        
        xsbhuu	 = self.BHUU(phi, kin1, kin2, kin3, kin4, F1, F2)
        xsiuu	 = self.IUU(phi, kin1, kin2, kin3, kin4, F1, F2, ReH, ReE, ReHtilde)

        f_pred = xsbhuu + xsiuu +  c1fit 

        #return torch.mean(torch.square((f_pred-f_true)/errs))
        return torch.mean(torch.square((f_pred-f_true)))

    def loss_MAE(self, kins, cffs, errs, f_true):
        phi, kin1, kin2, kin3, kin4, F1, F2 = kins
        ReH, ReE, ReHtilde, c1fit = cffs #output of network
        self.SetKinematics(kin1, kin2, kin3, kin4)
        xsbhuu   = self.BHUU(phi, kin1, kin2, kin3, kin4, F1, F2)
        xsiuu    = self.IUU(phi, kin1, kin2, kin3, kin4, F1, F2, ReH, ReE, ReHtilde)
        f_pred = xsbhuu + xsiuu +  c1fit 
        return torch.mean(torch.abs((f_pred-f_true)/f_true))
        
              
    def curve_fit(self, kins, cffs):
        phi, kin1, kin2, kin3, kin4, F1, F2 = kins
        ReH, ReE, ReHtilde, c1fit = cffs #output of network
        self.SetKinematics(kin1, kin2, kin3, kin4)
        xsbhuu   = self.BHUU(phi, kin1, kin2, kin3, kin4, F1, F2)
        xsiuu    = self.IUU(phi, kin1, kin2, kin3, kin4, F1, F2, ReH, ReE, ReHtilde)
        f_pred = xsbhuu + xsiuu +  c1fit 
        return f_pred

    def loss_chisq(self, kins, cffs, errs, f_true):
        phi, kin1, kin2, kin3, kin4, F1, F2 = kins
        ReH, ReE, ReHtilde, c1fit = cffs #output of network
        self.SetKinematics(kin1, kin2, kin3, kin4)
        xsbhuu   = self.BHUU(phi, kin1, kin2, kin3, kin4, F1, F2)
        xsiuu    = self.IUU(phi, kin1, kin2, kin3, kin4, F1, F2, ReH, ReE, ReHtilde)
        f_pred = xsbhuu + xsiuu +  c1fit 
        return torch.mean(torch.square((f_pred-f_true)/errs))
        #return torch.mean(torch.square((f_pred-f_true)))

    def loss_chisq_v2(self, kins, cffs, errs, f_true):
        phi, kin1, kin2, kin3, kin4, F1, F2 = kins
        ReH, ReE, ReHtilde, c1fit = cffs #output of network
        self.SetKinematics(kin1, kin2, kin3, kin4)
        xsbhuu   = self.BHUU(phi, kin1, kin2, kin3, kin4, F1, F2)
        xsiuu    = self.IUU(phi, kin1, kin2, kin3, kin4, F1, F2, ReH, ReE, ReHtilde)
        f_pred = xsbhuu + xsiuu +  c1fit 
        chisquare =  torch.mean(torch.square((f_pred-f_true)/errs))
        return torch.abs(1-chisquare)

    def loss_play(self, kins, cffs, errs, f_true):
        phi, kin1, kin2, kin3, kin4, F1, F2 = kins
        ReH, ReE, ReHtilde, c1fit = cffs #output of network
        self.SetKinematics(kin1, kin2, kin3, kin4)
        xsbhuu   = self.BHUU(phi, kin1, kin2, kin3, kin4, F1, F2)
        xsiuu    = self.IUU(phi, kin1, kin2, kin3, kin4, F1, F2, ReH, ReE, ReHtilde)
        f_pred = xsbhuu + xsiuu +  c1fit 
        chisquare =  torch.mean(torch.square((f_pred-f_true)/errs))
        mse = torch.mean(torch.square((f_pred-f_true)))
        return (chisquare + mse)
