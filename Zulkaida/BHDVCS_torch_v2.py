import torch
import numpy as np
import Lorentz_Vector as lv

class TBHDVCS(object):

    def __init__(self):
        self.ALP_INV = 137.0359998 # 1 / Electromagnetic Fine Structure Constant
        self.PI = 3.1415926535
        self.RAD = self.PI / 180.
        self.M = 0.938272 #Mass of the proton in GeV
        self.GeV2nb = .389379*1000000 # Conversion from GeV to NanoBar

        # 4-momentum vectors
        self.K = lv.LorentzVector()  
        self.KP = lv.LorentzVector()
        self.Q = lv.LorentzVector()
        self.QP  = lv.LorentzVector() 
        self.D  = lv.LorentzVector()
        self.p  = lv.LorentzVector()
        self.P  = lv.LorentzVector()

        # 4-vector products independent of cosphi
        self.kkp = 0 
        self.kq =0
        self.kp =0
        self.kpp=0
        

    def TProduct(self, v1, v2):
        return v1.Px()*v2.Px() + v1.Py()*v2.Py()

    def SetKinematics(self, _QQ, _x, _t, _k):

        self.QQ = _QQ #Q^2 value
        self.x = _x   #Bjorken x
        self.t = _t   #momentum transfer squared
        self.k = _k   #Electron Beam Energy
        self.M2 = self.M*self.M #Mass of the proton  squared in GeV
        #fractional energy of virtual photon
        self.y = self.QQ / ( 2. * self.M * self.k * self.x ) # From eq. (23) where gamma is substituted from eq (12c)
        #squared gamma variable ratio of virtuality to energy of virtual photon
        self.gg = 4. * self.M2 * self.x * self.x / self.QQ # This is gamma^2 [from eq. (12c)]
        #ratio of longitudinal to transverse virtual photon flux
        self.e = ( 1 - self.y - ( self.y * self.y * (self.gg / 4.) ) ) / ( 1. - self.y + (self.y * self.y / 2.) + ( self.y * self.y * (self.gg / 4.) ) ) # epsilon eq. (32)
        #Skewness parameter
        self.xi = 1. * self.x * ( ( 1. + self.t / ( 2. * self.QQ ) ) / ( 2. - self.x + self.x * self.t / self.QQ ) ) # skewness parameter eq. (12b) dnote: there is a minus sign on the write up that shouldn't be there
        #Minimum t value
        self.tmin = -( self.QQ * ( 1. - torch.sqrt( 1. + self.gg ) + self.gg / 2. ) ) / ( self.x * ( 1. - torch.sqrt( 1. + self.gg ) + self.gg / ( 2.* self.x ) ) ) # minimum t eq. (29)
        #Final Lepton energy
        self.kpr = self.k * ( 1. - self.y ) # k' from eq. (23)
        #outgoing photon energy
        self.qp = self.t / 2. / self.M + self.k - self.kpr #q' from eq. bellow to eq. (25) that has no numbering. Here nu = k - k' = k * y
        #Final proton Energy
        self.po = self.M - self.t / 2. / self.M # This is p'_0 from eq. (28b)
        self.pmag = np.sqrt( ( -1*self.t ) * ( 1. - (self.t / (4. * self.M *self.M ))) ) # p' magnitude from eq. (28b)
        #Angular Kinematics of outgoing photon
        self.cth = -1. / torch.sqrt( 1. + self.gg ) * ( 1. + self.gg / 2. * ( 1. + self.t / self.QQ ) / ( 1. + self.x * self.t / self.QQ ) ) # This is torch.cos(theta) eq. (26)
        self.theta = torch.acos(self.cth) # theta angle
        #print('Theta: ', self.theta)
        #Lepton Angle Kinematics of initial lepton
        self.sthl = torch.sqrt( self.gg ) / torch.sqrt( 1. + self.gg ) * ( torch.sqrt ( 1. - self.y - self.y * self.y * self.gg / 4. ) ) # sin(theta_l) from eq. (22a)
        self.cthl = -1. / torch.sqrt( 1. + self.gg ) * ( 1. + self.y * self.gg / 2. )  # torch.cos(theta_l) from eq. (22a)
        #ratio of momentum transfer to proton mass
        self.tau = -0.25 * self.t / self.M2

        # cosphi independent 4 - momenta vectors defined on eq. (21) -------------
        self.K.SetPxPyPzE( self.k * self.sthl, 0.0, self.k * self.cthl, self.k )
        self.KP.SetPxPyPzE( self.K[0], 0.0, self.k * ( self.cthl + self.y * torch.sqrt( 1. + self.gg ) ), self.kpr )
        self.Q = self.K - self.KP
        self.p.SetPxPyPzE(0.0, 0.0, 0.0, self.M)

        # Sets the Mandelstam variable s which is the center of mass energy
        self.s = (self.p + self.K) * (self.p + self.K)

        # The Gamma factor in front of the cross section
        self.Gamma = 1. / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / self.PI / 16. / ( self.s - self.M2 ) / ( self.s - self.M2 ) / torch.sqrt( 1. + self.gg ) / self.x

        # Defurne's Jacobian
        #self.jcob = 1./ ( 2. * self.M * self.x * self.K[3] ) * 2. * self.PI * 2.
        self.jcob = 2. * self.PI
        #print("Jacobian: ", self.jcob)
        #___________________________________________________________________________________
        
    def Set4VectorsPhiDep(self, cosphi) :

        # cosphi dependent 4 - momenta vectors defined on eq. (21) -------------

        self.QP.SetPxPyPzE(self.qp * torch.sin(self.theta) * cosphi, self.qp * torch.sin(self.theta) * torch.sqrt( 1 - cosphi*cosphi ), self.qp * torch.cos(self.theta), self.qp)
        self.D = self.Q - self.QP # delta vector eq. (12a)
        #print(self.D, "\n", self.Q, "\n", self.QP)
        self.pp = self.p + self.D # p' from eq. (21)
        self.P = self.p + self.pp
        self.P.SetPxPyPzE(.5*self.P.Px(), .5*self.P.Py(), .5*self.P.Pz(), .5*self.P.Pt())

        #____________________________________________________________________________________

    def Set4VectorProducts(self, cosphi) :

        # 4-vectors products (cosphi - independent)
        self.kkp  = self.K * self.KP   #(kk')
        self.kq   = self.K * self.Q    #(kq)
        self.kp   = self.K * self.p    #(pk)
        self.kpp  = self.KP * self.p   #(pk')

        # 4-vectors products (cosphi - dependent)
        self.kd   = self.K * self.D    #(kd)
        self.kpd  = self.KP * self.D   #(k'd)
        self.kP   = self.K * self.P    #(kP)
        self.kpP  = self.KP * self.P   #(k'P)
        self.kqp  = self.K * self.QP   #(kq')
        self.kpqp = self.KP * self.QP  #(k'q')
        self.dd   = self.D * self.D    #(dd)
        self.Pq   = self.P * self.Q    #(Pq)
        self.Pqp  = self.P * self.QP   #(Pq')
        self.qd   = self.Q * self.D    #(qd)
        self.qpd  = self.QP * self.D   #(q'd)

        # #Transverse vector products defined after eq.(241c) -----------------------
        self.kk_T   = self.TProduct(self.K,self.K)
        self.kkp_T  = self.kk_T
        self.kqp_T  = self.TProduct(self.K,self.QP)
        self.kd_T   = -1.* self.kqp_T
        self.dd_T   = self.TProduct(self.D,self.D)
        self.kpqp_T = self.kqp_T
        self.kP_T   = self.TProduct(self.K,self.P)
        self.kpP_T  = self.TProduct(self.KP,self.P)
        self.qpP_T  = self.TProduct(self.QP,self.P)
        self.kpd_T  = -1.* self.kqp_T
        self.qpd_T  = -1. * self.dd_T
        
        #____________________________________________________________________________________

        #Expressions that appear in the interference coefficient calculations
        self.Dplus   = .5 / (self.kpqp) - .5 / (self.kqp)
        self.Dminus  = -.5 / (self.kpqp) - .5 / (self.kqp)

    def GetBHUUxs(self, cosphi, F1, F2) :

        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        # Coefficients of the BH unpolarized structure function FUUBH
        self.AUUBH = ( (8. * self.M2) / (self.t * self.kqp * self.kpqp) ) * ( (4. * self.tau * (self.kP * self.kP + self.kpP * self.kpP) ) - ( (self.tau + 1.) * (self.kd * self.kd + self.kpd * self.kpd) ) )
        self.BUUBH = ( (16. * self.M2) / (self.t* self.kqp * self.kpqp) ) * (self.kd * self.kd + self.kpd * self.kpd)

        # Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
        # I multiply by 2 because I think Auu and Buu are missing a factor 2
        self.con_AUUBH = self.AUUBH * self.GeV2nb * self.jcob
        self.con_BUUBH = self.BUUBH * self.GeV2nb * self.jcob

        # Unpolarized Coefficients multiplied by the Form Factors
        self.bhAUU = (self.Gamma/self.t) * self.con_AUUBH * ( F1 * F1 + self.tau * F2 * F2 )
        self.bhBUU = (self.Gamma/self.t) * self.con_BUUBH * ( self.tau * ( F1 + F2 ) * ( F1 + F2 ) ) 

        # Unpolarized BH cross section
        self.xbhUU = self.bhAUU + self.bhBUU

        return self.xbhUU
        
        #____________________________________________________________________________________

    def GetIUUxs(self, cosphi, F1, F2, ReH, ReE, ReHtilde) :
        
        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        self.AUUI = -4. * cosphi * ( self.Dplus * ( ( self.kqp_T - 2. * self.kk_T - 2. * self.kqp ) * self.kpP + ( 2. * self.kpqp - 2. * self.kkp_T - self.kpqp_T ) * self.kP +         	                 self.kpqp * self.kP_T + self.kqp * self.kpP_T - 2.*self.kkp * self.kP_T ) - self.Dminus * ( ( 2. * self.kkp - self.kpqp_T - self.kkp_T ) * self.Pqp + 2. * self.kkp * self.qpP_T -    	              self.kpqp * self.kP_T - self.kqp * self.kpP_T ) ) 

        self.BUUI = -2. * self.xi * cosphi * ( self.Dplus * ( ( self.kqp_T - 2. * self.kk_T - 2. * self.kqp ) * self.kpd + ( 2. * self.kpqp - 2. * self.kkp_T - self.kpqp_T ) * self.kd   	                    + self.kpqp * self.kd_T + self.kqp * self.kpd_T - 2.*self.kkp * self.kd_T ) - self.Dminus * ( ( 2. * self.kkp - self.kpqp_T - self.kkp_T ) * self.qpd + 2. * self.kkp * self.qpd_T -  	                self.kpqp * self.kd_T - self.kqp * self.kpd_T ) )

        self.CUUI = -2. * cosphi * ( self.Dplus * ( 2. * self.kkp * self.kd_T - self.kpqp * self.kd_T - self.kqp * self.kpd_T + 4. * self.xi * self.kkp * self.kP_T - 2. * self.xi *   	                self.kpqp * self.kP_T - 2. * self.xi * self.kqp * self.kpP_T ) -
                    self.Dminus * ( self.kkp * self.qpd_T - self.kpqp * self.kd_T - self.kqp * self.kpd_T + 2. * self.xi * self.kkp * self.qpP_T - 2. * self.xi * self.kpqp * self.kP_T       	   		            - 2. * self.xi * self.kqp  * self.kpP_T ) )

        # Convert Unpolarized Coefficients to nano-barn and use Defurne's Jacobian
        self.con_AUUI = self.AUUI * self.GeV2nb * self.jcob
        self.con_BUUI = self.BUUI * self.GeV2nb * self.jcob
        self.con_CUUI = self.CUUI * self.GeV2nb * self.jcob

        #Unpolarized Coefficients multiplied by the Form Factors
        self.iAUU = (self.Gamma/(torch.abs(self.t) * self.QQ)) * self.con_AUUI * ( F1 * ReH + self.tau * F2 * ReE )
        self.iBUU = (self.Gamma/(torch.abs(self.t) * self.QQ)) * self.con_BUUI * ( F1 + F2 ) * ( ReH + ReE )
        self.iCUU = (self.Gamma/(torch.abs(self.t) * self.QQ)) * self.con_CUUI * ( F1 + F2 ) * ReHtilde

        #Unpolarized BH-DVCS interference cross section
        self.xIUU = self.iAUU + self.iBUU + self.iCUU

        return -1. * self.xIUU
    
    def TotalUUXS(self, angle, par):
        cosphi = angle[0]
	    # Set QQ, xB, t and k and calculate 4-vector products
        #print(par)
        self.SetKinematics(par[0], par[1], par[2], par[3] )
        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        xsbhuu	 = self.GetBHUUxs(cosphi, par[4], par[5])
        xsiuu	 = self.GetIUUxs(cosphi, par[4], par[5], par[6], par[7], par[8])
        tot_sigma_uu = xsbhuu + xsiuu + par[9] # Constant added to account for DVCS contribution
        #print(xsbhuu, " ", xsiuu, " ", tot_sigma_uu)
        return tot_sigma_uu
    
    def TotalUUXS_curve_fit(self, x, ReH, ReE, ReHT):
        cosphi, kin1, kin2, kin3, kin4, F1, F2, const = x
	    # Set QQ, xB, t and k and calculate 4-vector products
        #print(par)
        self.SetKinematics(kin1, kin2, kin3, kin4)
        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        xsbhuu	 = self.GetBHUUxs(cosphi, F1, F2)
        xsiuu	 = self.GetIUUxs(cosphi, F1, F2, ReH, ReE, ReHT) 

        tot_sigma_uu = xsbhuu + xsiuu +  const # Constant added to account for DVCS contribution
        return tot_sigma_uu

    def TotalUUXS_curve_fit2(self, cosphi, qq, xb, t, k, F1, F2, const, ReH, ReE, ReHT):
	    # Set QQ, xB, t and k and calculate 4-vector products
        #print(par)
        self.SetKinematics(qq, xb, t, k)
        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        xsbhuu	 = self.GetBHUUxs(cosphi, F1, F2)
        xsiuu	 = self.GetIUUxs(cosphi, F1, F2, ReH, ReE, ReHT) 

        tot_sigma_uu = xsbhuu + xsiuu +  const # Constant added to account for DVCS contribution
        return tot_sigma_uu

    def TotalUUXS_curve_fit3(self, x1, x2):
        cosphi, kin1, kin2, kin3, kin4, F1, F2, const = x1
        ReH, ReE, ReHT = x2
	    # Set QQ, xB, t and k and calculate 4-vector products
        #print(par)
        self.SetKinematics(kin1, kin2, kin3, kin4)
        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        xsbhuu	 = self.GetBHUUxs(cosphi, F1, F2)
        xsiuu	 = self.GetIUUxs(cosphi, F1, F2, ReH, ReE, ReHT) 

        tot_sigma_uu = xsbhuu + xsiuu +  const # Constant added to account for DVCS contribution
        return tot_sigma_uu

    def loss_MSE(self, kins, cffs, errs, f_true):
        cosphi, kin1, kin2, kin3, kin4, F1, F2, const = kins
        ReH, ReE, ReHT = cffs #output of network

        self.SetKinematics(kin1, kin2, kin3, kin4)
        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        xsbhuu	 = self.GetBHUUxs(cosphi, F1, F2)
        xsiuu	 = self.GetIUUxs(cosphi, F1, F2, ReH, ReE, ReHT)

        f_pred = xsbhuu + xsiuu +  const

        return torch.mean(torch.square((f_pred-f_true)/errs))

    def loss_ABS(self, kins, cffs, errs, f_true):
        cosphi, kin1, kin2, kin3, kin4, F1, F2, const = kins
        ReH, ReE, ReHT = cffs #output of network

        self.SetKinematics(kin1, kin2, kin3, kin4)
        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        xsbhuu	 = self.GetBHUUxs(cosphi, F1, F2)
        xsiuu	 = self.GetIUUxs(cosphi, F1, F2, ReH, ReE, ReHT)

        f_pred = xsbhuu + xsiuu +  const

        return torch.mean((torch.abs(f_pred-f_true)/errs))

    def loss_PH(self, kins, cffs, errs, f_true, delta): #Psuedo Huber Loss
        cosphi, kin1, kin2, kin3, kin4, F1, F2, const = kins
        ReH, ReE, ReHT = cffs #output of network

        self.SetKinematics(kin1, kin2, kin3, kin4)
        self.Set4VectorsPhiDep(cosphi)
        self.Set4VectorProducts(cosphi)

        xsbhuu	 = self.GetBHUUxs(cosphi, F1, F2)
        xsiuu	 = self.GetIUUxs(cosphi, F1, F2, ReH, ReE, ReHT)

        f_pred = xsbhuu + xsiuu + const

        dy = torch.mean(f_true - f_pred)
        if dy < delta:
            return self.loss_MSE(kins, cffs, errs, f_true)
        else:
            return self.loss_ABS(kins, cffs, errs, f_true)
