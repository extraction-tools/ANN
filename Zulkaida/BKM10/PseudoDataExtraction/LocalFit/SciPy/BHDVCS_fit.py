import numpy as np

class DVCSFIT(object):

    def __init__(self):
        self.ALP_INV = 137.0359998 # 1 / Electromagnetic Fine Structure Constant
        self.PI = 3.1415926535
        self.RAD = self.PI / 180.
        self.M = 0.938272 #Mass of the proton in GeV
        self.GeV2nb = .389379*1000000 # Conversion from GeV to NanoBar
        self.M2 = self.M*self.M #Mass of the proton  squared in GeV
        

      
    def SetKinematics(self, kinematics):
        _QQ, _x, _t, _k = kinematics        

        self.QQ = _QQ #Q^2 value
        self.x = _x   #Bjorken x
        self.t = _t   #momentum transfer squared
        self.k = _k   #Electron Beam Energy
        
        self.ee = 4. * self.M2 * self.x * self.x / self.QQ # epsilon squared
        self.y = np.sqrt(self.QQ) / ( np.sqrt(self.ee) * self.k )  # lepton energy fraction
        self.xi = self.x * ( 1. + self.t / 2. / self.QQ ) / ( 2. - self.x + self.x * self.t / self.QQ ); # Generalized Bjorken variable
        self.Gamma = self.x * self.y * self.y / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / 8. / self.QQ / self.QQ / np.sqrt( 1. + self.ee ) # factor in front of the cross section, eq. (22)
        self.s = 2. * self.M * self.k + self.M2
        self.tmin = - self.QQ * ( 2. * ( 1. - self.x ) * ( 1. - np.sqrt(1. + self.ee) ) + self.ee ) / ( 4. * self.x * ( 1. - self.x ) + self.ee ) # eq. (31)
        self.Ktilde_10 = np.sqrt( self.tmin - self.t ) * np.sqrt( ( 1. - self.x ) * np.sqrt( 1. + self.ee ) + ( ( self.t - self.tmin ) * ( self.ee + 4. * self.x * ( 1. - self.x ) ) / 4. / self.QQ ) ) * np.sqrt( 1. - self.y - self.y * self.y * self.ee / 4. ) / np.sqrt( 1. - self.y + self.y * self.y * self.ee / 4.) # K tilde from 2010 paper
        self.K = np.sqrt( 1. - self.y + self.y * self.y * self.ee / 4.) * self.Ktilde_10 / np.sqrt(self.QQ)
        #___________________________________________________________________________________



    def BHLeptonPropagators(self,kinematics ,phi ):
        self.SetKinematics(kinematics)

	#KD 4-vector product (phi-dependent)
        self.KD = - self.QQ / ( 2. * self.y * ( 1. + self.ee ) ) * ( 1. + 2. * self.K * np.cos( self.PI - ( phi * self.RAD ) ) - self.t / self.QQ * ( 1. - self.x * ( 2. - self.y ) + self.y * self.ee / 2. ) + self.y * self.ee / 2.  ) # eq. (29)

  	# lepton BH propagators P1 and P2 (contaminating phi-dependence)
        self.P1 = 1. + 2. * self.KD / self.QQ
        self.P2 = self.t / self.QQ - 2. * self.KD / self.QQ
        
    

    def BHUU(self, kinematics, phi, F1, F2) :

        self.BHLeptonPropagators(kinematics, phi)

        #BH unpolarized Fourier harmonics eqs. (35 - 37)
        
        self.c0_BH = 8. * self.K * self.K * ( ( 2. + 3. * self.ee ) * ( self.QQ / self.t ) * ( F1 * F1  - F2 * F2 * self.t / ( 4. * self.M2 ) ) + 2. * self.x * self.x * ( F1 + F2 ) * ( F1 + F2 ) ) + ( 2. - self.y ) * ( 2. - self.y ) * ( ( 2. + self.ee ) * ( ( 4. * self.x * self.x * self.M2 / self.t ) * ( 1. + self.t / self.QQ ) * ( 1. + self.t / self.QQ ) + 4. * ( 1. - self.x ) * ( 1. + self.x * self.t / self.QQ ) ) * ( F1 * F1 - F2 * F2 * self.t / ( 4. * self.M2 ) ) + 4. * self.x * self.x * ( self.x + ( 1. - self.x + self.ee / 2. ) * ( 1. - self.t / self.QQ ) * ( 1. - self.t / self.QQ ) - self.x * ( 1. - 2. * self.x ) * self.t * self.t / ( self.QQ * self.QQ ) ) * ( F1 + F2 ) * ( F1 + F2 ) ) + 8. * ( 1. + self.ee ) * ( 1. - self.y - self.ee * self.y * self.y / 4. ) * ( 2. * self.ee * ( 1. - self.t / ( 4. * self.M2 ) ) * ( F1 * F1 - F2 * F2 * self.t / ( 4. * self.M2 ) ) - self.x * self.x * ( 1. - self.t / self.QQ ) * ( 1. - self.t / self.QQ ) * ( F1 + F2 ) * ( F1 + F2 ) )

        self.c1_BH = 8. * self.K * ( 2. - self.y ) * ( ( 4. * self.x * self.x * self.M2 / self.t - 2. * self.x - self.ee ) * ( F1 * F1 - F2 * F2 * self.t / ( 4. * self.M2 ) ) + 2. * self.x * self.x * ( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ ) * ( F1 + F2 ) * ( F1 + F2 ) )

        self.c2_BH = 8. * self.x * self.x * self.K * self.K * ( ( 4. * self.M2 / self.t ) * ( F1 * F1 - F2 * F2 * self.t / ( 4. * self.M2 ) ) + 2. * ( F1 + F2 ) * ( F1 + F2 ) )

        #BH squared amplitude eq (25) divided by e^6
        self.Amp2_BH = 1. / ( self.x * self.x * self.y * self.y * ( 1. + self.ee ) * ( 1. + self.ee ) * self.t * self.P1 * self.P2 ) * ( self.c0_BH + self.c1_BH * np.cos( self.PI - (phi * self.RAD) ) + self.c2_BH * np.cos( 2. * ( self.PI - ( phi * self.RAD ) ) )  )

        self.Amp2_BH = self.GeV2nb * self.Amp2_BH # convertion to nb

        return self.Gamma * self.Amp2_BH

 
    def IUU(self, kinematics, phi, F1, F2, ReH, ReE, ReHtilde, twist) :
        
        # Get BH propagators and set the kinematics
        self.BHLeptonPropagators(kinematics, phi)
       
        #Get A_UU_I, B_UU_I and C_UU_I interference coefficients
        self.ABC_UU_I_10(kinematics, phi, twist);

        #BH-DVCS interference squared amplitude
        self.I_10 = 1. / ( self.x * self.y * self.y * self.y * self.t * self.P1 * self.P2 ) * ( self.A_U_I * ( F1 * ReH - self.t / 4. / self.M2 * F2 * ReE ) + self.B_U_I * ( F1 + F2 ) * ( ReH + ReE ) + self.C_U_I * ( F1 + F2 ) * ReHtilde )

        self.I_10 = self.GeV2nb * self.I_10 # convertion to nb

        return  self.Gamma * self.I_10


    def ABC_UU_I_10(self, kinematics, phi, twist) : #Get A_UU_I, B_UU_I and C_UU_I interference coefficients BKM10
        self.SetKinematics(kinematics)

        #F_eff = f * F
        if twist == "t2": 
           self.f = 0 # F_eff = 0 ( pure twist 2)
        if twist == "t3":
           self.f = - 2. * self.xi / ( 1. + self.xi )
        if twist == "t3ww": 
           self.f = 2. / ( 1. + self.xi )

        # Interference coefficients  (BKM10 Appendix A.1)
        # n = 0 -----------------------------------------
        # helicity - conserving (F)
        self.C_110 = - 4. * ( 2. - self.y ) * ( 1. + np.sqrt( 1 + self.ee ) ) / np.power(( 1. + self.ee ), 2) * ( self.Ktilde_10 * self.Ktilde_10 * ( 2. - self.y ) * ( 2. - self.y ) / self.QQ / np.sqrt( 1 + self.ee )
            + self.t / self.QQ * ( 1. - self.y - self.ee / 4. * self.y * self.y ) * ( 2. - self.x ) * ( 1. + ( 2. * self.x * ( 2. - self.x + ( np.sqrt( 1. + self.ee ) - 1. ) / 2. + self.ee / 2. / self.x ) * self.t / self.QQ + self.ee ) / ( 2. - self.x ) / ( 1. + np.sqrt( 1. + self.ee ) ) ) )
        self.C_110_V = 8. * ( 2. - self.y ) / np.power(( 1. + self.ee ), 2) * self.x * self.t / self.QQ * ( ( 2. - self.y ) * ( 2. - self.y ) / np.sqrt( 1. + self.ee ) * self.Ktilde_10 * self.Ktilde_10 / self.QQ
              + ( 1. - self.y - self.ee / 4. * self.y * self.y ) * ( 1. + np.sqrt( 1. + self.ee ) ) / 2. * ( 1. + self.t / self.QQ ) * ( 1. + ( np.sqrt ( 1. + self.ee ) - 1. + 2. * self.x ) / ( 1. + np.sqrt( 1. + self.ee ) ) * self.t / self.QQ ) )
        self.C_110_A = 8. * ( 2. - self.y ) / np.power(( 1. + self.ee ), 2) * self.t / self.QQ * ( ( 2. - self.y ) * ( 2. - self.y ) / np.sqrt( 1. + self.ee ) * self.Ktilde_10 * self.Ktilde_10 / self.QQ * ( 1. + np.sqrt( 1. + self.ee ) - 2. * self.x ) / 2.
              + ( 1. - self.y - self.ee / 4. * self.y * self.y ) * ( ( 1. + np.sqrt( 1. + self.ee ) ) / 2. * ( 1. + np.sqrt( 1. + self.ee ) - self.x + ( np.sqrt( 1. + self.ee ) - 1. + self.x * ( 3. + np.sqrt( 1. + self.ee ) - 2. * self.x ) / ( 1. + np.sqrt( 1. + self.ee ) ) )
              * self.t / self.QQ ) - 2. * self.Ktilde_10 * self.Ktilde_10 / self.QQ ) )
        # helicity - changing (F_eff)
        self.C_010 = 12. * np.sqrt(2.) * self.K * ( 2. - self.y ) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * ( self.ee + ( 2. - 6. * self.x - self.ee ) / 3. * self.t / self.QQ )
        self.C_010_V = 24. * np.sqrt(2.) * self.K * ( 2. - self.y ) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * self.x * self.t / self.QQ * ( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ )
        self.C_010_A = 4. * np.sqrt(2.) * self.K * ( 2. - self.y ) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * self.t / self.QQ * ( 8. - 6. * self.x + 5. * self.ee ) * ( 1. - self.t / self.QQ * ( ( 2. - 12 * self.x * ( 1. - self.x ) - self.ee )
              / ( 8. - 6. * self.x + 5. * self.ee ) ) )
        # n = 1 -----------------------------------------
        # helicity - conserving (F)
        self.C_111 = -16. * self.K * ( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * ( ( 1. + ( 1. - self.x ) * ( np.sqrt( 1 + self.ee ) - 1. ) / 2. / self.x + self.ee / 4. / self.x ) * self.x * self.t / self.QQ - 3. * self.ee / 4. ) - 4. * self.K * ( 2. - 2. * self.y + self.y * self.y + self.ee / 2. * self.y * self.y ) * ( 1. + np.sqrt( 1 + self.ee ) - self.ee ) / np.power(np.sqrt( 1. + self.ee ), 5) * ( 1. - ( 1. - 3. * self.x ) * self.t / self.QQ + ( 1. - np.sqrt( 1 + self.ee ) + 3. * self.ee ) / ( 1. + np.sqrt( 1 + self.ee ) - self.ee ) * self.x * self.t / self.QQ ) 
        self.C_111_V = 16. * self.K / np.power(np.sqrt( 1. + self.ee ), 5) * self.x * self.t / self.QQ * ( ( 2. - self.y ) * ( 2. - self.y ) * ( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ ) + ( 1. - self.y - self.ee / 4. * self.y * self.y )
              * ( 1. + np.sqrt( 1. + self.ee ) - 2. * self.x ) / 2. * ( self.t - self.tmin ) / self.QQ )
        self.C_111_A = -16. * self.K / np.power(( 1. + self.ee ), 2) * self.t / self.QQ * ( ( 1. - self.y - self.ee / 4. * self.y * self.y ) * ( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ + ( 4. * self.x * ( 1. - self.x ) + self.ee ) / 4. / np.sqrt( 1. + self.ee ) * ( self.t - self.tmin ) / self.QQ )
              - np.power(( 2. - self.y ), 2) * ( 1. - self.x / 2. + ( 1. + np.sqrt( 1. + self.ee ) - 2. * self.x ) / 4. * ( 1. - self.t / self.QQ ) + ( 4. * self.x * ( 1. - self.x ) + self.ee ) / 2. / np.sqrt( 1. + self.ee ) * ( self.t - self.tmin ) / self.QQ ) )
        # helicity - changing (F_eff)
        self.C_011 = 8. * np.sqrt(2.) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(( 1. + self.ee ), 2) * ( np.power(( 2. - self.y ), 2) * ( self.t - self.tmin ) / self.QQ * ( 1. - self.x + ( ( 1. - self.x ) * self.x + self.ee / 4. ) / np.sqrt( 1. + self.ee ) * ( self.t - self.tmin ) / self.QQ )
            + ( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.sqrt( 1 + self.ee ) * ( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ ) * ( self.ee - 2. * ( 1. + self.ee / 2. / self.x ) * self.x * self.t / self.QQ ) )
        self.C_011_V = 16. * np.sqrt(2.) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * self.x * self.t / self.QQ * ( np.power( self.Ktilde_10 * ( 2. - self.y ), 2) / self.QQ + np.power(( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ ), 2) * ( 1. - self.y - self.ee / 4. * self.y * self.y ) )
        self.C_011_A = 8. * np.sqrt(2.) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * self.t / self.QQ * ( np.power( self.Ktilde_10 * ( 2. - self.y ), 2) * ( 1. - 2. * self.x ) / self.QQ + ( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ )
              * ( 1. - self.y - self.ee / 4. * self.y * self.y ) * ( 4. - 2. * self.x + 3. * self.ee + self.t / self.QQ * ( 4. * self.x * ( 1. - self.x ) + self.ee ) ) )
       # n = 2 -----------------------------------------
       # helicity - conserving (F)
        self.C_112 = 8. * ( 2. - self.y ) * ( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(( 1. + self.ee ), 2) * ( 2. * self.ee / np.sqrt( 1. + self.ee ) / ( 1. + np.sqrt( 1. + self.ee ) ) * np.power(self.Ktilde_10, 2) / self.QQ + self.x * self.t * ( self.t - self.tmin ) / self.QQ / self.QQ * ( 1. - self.x - ( np.sqrt( 1. + self.ee ) - 1. ) / 2. + self.ee / 2. / self.x ))
        self.C_112_V = 8. * ( 2. - self.y ) * ( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(( 1. + self.ee ), 2) * self.x * self.t / self.QQ * ( 4. * np.power(self.Ktilde_10, 2) / np.sqrt( 1. + self.ee ) / self.QQ + ( 1. + np.sqrt( 1. + self.ee ) - 2. * self.x ) / 2. * ( 1. + self.t / self.QQ ) * ( self.t - self.tmin ) / self.QQ )
        self.C_112_A = 4. * ( 2. - self.y ) * ( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(( 1. + self.ee ), 2) * self.t / self.QQ * ( 4. * ( 1. - 2. * self.x ) * np.power(self.Ktilde_10, 2) / np.sqrt( 1. + self.ee ) / self.QQ - ( 3. -  np.sqrt( 1. + self.ee ) - 2. * self.x + self.ee / self.x ) * self.x * ( self.t - self.tmin ) / self.QQ )
       # helicity - changing (F_eff)
        self.C_012 = -8. * np.sqrt(2.) * self.K * ( 2. - self.y ) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * ( 1. + self.ee / 2. ) * ( 1. + ( 1. + self.ee / 2. / self.x ) / ( 1. + self.ee / 2. ) * self.x * self.t / self.QQ )
        self.C_012_V = 8. * np.sqrt(2.) * self.K * ( 2. - self.y ) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * self.x * self.t / self.QQ * ( 1. - ( 1. - 2. * self.x ) * self.t / self.QQ )
        self.C_012_A = 8. * np.sqrt(2.) * self.K * ( 2. - self.y ) * np.sqrt( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(( 1. + self.ee ), 2) * self.t / self.QQ * ( 1. - self.x + ( self.t - self.tmin ) / 2. / self.QQ * ( 4. * self.x * ( 1. - self.x ) + self.ee ) / np.sqrt( 1. + self.ee ) )
       # n = 3 -----------------------------------------
       # helicity - conserving (F)
        self.C_113 = -8. * self.K * ( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * ( np.sqrt( 1. + self.ee ) - 1. ) * ( ( 1. - self.x ) * self.t / self.QQ + ( np.sqrt( 1. + self.ee ) - 1. ) / 2. * ( 1. + self.t / self.QQ ) )
        self.C_113_V = -8. * self.K * ( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * self.x * self.t / self.QQ * ( np.sqrt( 1. + self.ee ) - 1. + ( 1. + np.sqrt( 1. + self.ee ) - 2. * self.x ) * self.t / self.QQ )
        self.C_113_A = 16. * self.K * ( 1. - self.y - self.ee / 4. * self.y * self.y ) / np.power(np.sqrt( 1. + self.ee ), 5) * self.t * ( self.t - self.tmin ) / self.QQ / self.QQ * ( self.x * ( 1. - self.x ) + self.ee / 4. )

       # A_U_I, B_U_I and C_U_I
        self.A_U_I = self.C_110 + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * self.C_010 + ( self.C_111 + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * self.C_011 ) * np.cos( self.PI - (phi * self.RAD) ) + ( self.C_112 + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * self.C_012 ) * np.cos( 2. * ( self.PI - (phi *self. RAD) ) ) + self.C_113 * np.cos( 3. * ( self.PI - (phi * self.RAD) ) )
        self.B_U_I = self.xi / ( 1. + self.t / 2. / self.QQ ) * ( self.C_110_V + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * self.C_010_V + ( self.C_111_V + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * self.C_011_V ) * np.cos( self.PI - (phi * self.RAD) ) + ( self.C_112_V + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * self.C_012_V ) * np.cos( 2. * ( self.PI - (phi * self.RAD) ) ) + self.C_113_V * np.cos( 3. * ( self.PI - (phi * self.RAD) ) ) )
        self.C_U_I = self.xi / ( 1. + self.t / 2. / self.QQ ) * ( self.C_110 + self.C_110_A + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * ( self.C_010 + self.C_010_A ) + ( self.C_111 + self.C_111_A + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * ( self.C_011 + self.C_011_A ) ) * np.cos( self.PI - (phi * self.RAD) ) + ( self.C_112 + self.C_112_A + np.sqrt(2) / ( 2. - self.x ) * self.Ktilde_10 / np.sqrt(self.QQ) * self.f * ( self.C_012 + self.C_012_A ) ) * np.cos( 2. * ( self.PI - (phi * self.RAD) ) ) + ( self.C_113 + self.C_113_A ) * np.cos( 3. * ( self.PI - (phi * self.RAD) ) ) )

    
    
    def curve_fit2(self, kins, ReH, ReE, ReHtilde, c0fit, c1fit):
        phi, kin1, kin2, kin3, kin4, F1, F2 = kins
        kinematics = (kin1, kin2, kin3, kin4)
        self.SetKinematics(kinematics)
        xsbhuu   = self.BHUU(kinematics, phi, F1, F2)
        xsiuu    = self.IUU(kinematics, phi, F1, F2, ReH, ReE, ReHtilde, "t3")
        f_pred = xsbhuu + xsiuu + c0fit + c1fit * np.cos( self.PI - (phi * self.RAD) )
        return f_pred


    def plot_fit(self, kins, cffs):
        phi, kin1, kin2, kin3, kin4, F1, F2 = kins
        ReH, ReE, ReHtilde, c0fit, c1fit = cffs #output of network
        kinematics = [kin1, kin2, kin3, kin4]
        self.SetKinematics(kinematics)
        xsbhuu   = self.BHUU(kinematics, phi, F1, F2)
        xsiuu    = self.IUU(kinematics, phi, F1, F2, ReH, ReE, ReHtilde, "t3")
        f_pred = xsbhuu + xsiuu + c0fit + c1fit * np.cos( self.PI - (phi * self.RAD) )
        return f_pred

