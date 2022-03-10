import tensorflow as tf
import numpy as np


class TotalFLayer(tf.keras.layers.Layer):
    
    def __init__(self):
        super(TotalFLayer, self).__init__(dtype='float32')
        self.F=BHDVCStf()
        
    def call(self, inputs):
        return self.F.curve_fit_array(inputs[:, 0:7], inputs[:, 7:11])

class BHDVCStf(tf.Module):
    def __init__(self, name=None):
      super(BHDVCStf, self).__init__(name=name)
      self.ALP_INV = tf.constant(137.0359998) # 1 / Electromagnetic Fine Structure Constant
      self.PI = tf.constant(3.1415926535)
      self.RAD = tf.constant(3.1415926535 / 180.)
      self.M = tf.constant(0.938272) #Mass of the proton in GeV
      self.GeV2nb = tf.constant(.389379*1000000) # Conversion from GeV to NanoBar
      self.M2 = tf.constant(0.938272*0.938272) #Mass of the proton  squared in GeV
        
    @tf.function  
    def setKinematics(self, phi, QQ, x, t, k):
        ee = 4. * self.M2  * x * x / QQ # epsilon squared
        y = tf.sqrt(QQ) / ( tf.sqrt(ee) * k )  # lepton energy fraction
        # xi = x * ( 1. + t / 2. / QQ ) / ( 2. - x + x * t / QQ ); # Generalized Bjorken variable
        Gamma1 = x * y * y / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / 8. / QQ / QQ / tf.sqrt( 1. + ee ) # factor in front of the cross section, eq. (22)
        # s = 2. * self.M * k + self.M2
        tmin = - QQ * ( 2. * ( 1. - x ) * ( 1. - tf.sqrt(1. + ee) ) + ee ) / ( 4. * x * ( 1. - x ) + ee ) # eq. (31)
        k2 = - ( t / QQ ) * ( 1. - x ) * ( 1. - y - y * y * ee / 4.) * ( 1. - tmin / t ) * ( tf.sqrt(1. + ee) + ( ( 4. * x * ( 1. - x ) + ee ) / ( 4. * ( 1. - x ) ) ) * ( ( t - tmin ) / QQ )  ) # eq. (30)_____________________________________________
        return ee, y, Gamma1, k2

    @tf.function
    def BHLeptonPropagators(self, phi, QQ, x, t, k, ee, y, k2):
	      #K*D 4-vector product (phi-dependent)
        KD = - QQ / ( 2. * y * ( 1. + ee ) ) * ( 1. + 2. * tf.sqrt(k2) * tf.cos( self.PI - ( phi * self.RAD ) ) - t / QQ * ( 1. - x * ( 2. - y ) + y * ee / 2. ) + y * ee / 2.  ) #eq. (29)
  	    # lepton BH propagators P1 and P2 (contaminating phi-dependence)
        P1 = 1. + 2. * KD / QQ
        P2 = t / QQ - 2. * KD / QQ
        return P1, P2
        
    
    @tf.function
    def BHUU(self, phi, QQ, x, t, k, F1, F2, ee, y, Gamma1, k2, P1, P2) :
        #BH unpolarized Fourier harmonics eqs. (35 - 37)
        c0_BH = 8. * k2 * ( ( 2. + 3. * ee ) * ( QQ / t ) * ( F1 * F1  - F2 * F2 * t / ( 4. * self.M2 ) ) + 2. * x * x * ( F1 + F2 ) * ( F1 + F2 ) ) + ( 2. - y ) * ( 2. - y ) * ( ( 2. + ee ) * ( ( 4. * x * x * self.M2 / t ) * ( 1. + t / QQ ) * ( 1. + t / QQ ) + 4. * ( 1. - x ) * ( 1. + x * t / QQ ) ) * ( F1 * F1 - F2 * F2 * t / ( 4. * self.M2 ) ) + 4. * x * x * ( x + ( 1. - x + ee / 2. ) * ( 1. - t / QQ ) * ( 1. - t / QQ ) - x * ( 1. - 2. * x ) * t * t / ( QQ * QQ ) ) * ( F1 + F2 ) * ( F1 + F2 ) ) + 8. * ( 1. + ee ) * ( 1. - y - ee * y * y / 4. ) * ( 2. * ee * ( 1. - t / ( 4. * self.M2 ) ) * ( F1 * F1 - F2 * F2 * t / ( 4. * self.M2 ) ) - x * x * ( 1. - t / QQ ) * ( 1. - t / QQ ) * ( F1 + F2 ) * ( F1 + F2 ) )
        c1_BH = 8. * tf.sqrt(k2) * ( 2. - y ) * ( ( 4. * x * x * self.M2 / t - 2. * x - ee ) * ( F1 * F1 - F2 * F2 * t / ( 4. * self.M2 ) ) + 2. * x * x * ( 1. - ( 1. - 2. * x ) * t / QQ ) * ( F1 + F2 ) * ( F1 + F2 ) )
        c2_BH = 8. * x * x * k2 * ( ( 4. * self.M2 / t ) * ( F1 * F1 - F2 * F2 * t / ( 4. * self.M2 ) ) + 2. * ( F1 + F2 ) * ( F1 + F2 ) )

  	#BH squared amplitude eq (25) divided by e^6
        Amp2_BH = 1. / ( x * x * y * y * ( 1. + ee ) * ( 1. + ee ) * t * P1 * P2 ) * ( c0_BH + c1_BH * tf.cos( self.PI - (phi * self.RAD) ) + c2_BH * tf.cos( 2. * ( self.PI - ( phi * self.RAD ) ) )  )
        Amp2_BH = self.GeV2nb * Amp2_BH #convertion to nb

  	#return self.dsigma_BH = Gamma1 * Amp2_BH
        dsigma_BH = Gamma1 * Amp2_BH
        return dsigma_BH
        
    @tf.function   
    def IUU(self, phi, QQ, x, t, k, F1, F2, ReH, ReE, ReHtilde, y, Gamma1, k2, P1, P2) :        
        A = - 8. * k2 * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * ( 2. - y ) * ( 1. - y ) * ( 2. - x ) * t / QQ - 8. * tf.sqrt(k2) * ( 2. - 2. * y + y * y ) * tf.cos( self.PI - (phi * self.RAD) )
        B = 8. * x * x * ( 2. - y ) * (1 - y ) / ( 2. - x ) * t / QQ
        C =  x / ( 2. - x ) * ( - 8. * k2 * ( 2. - y ) * ( 2. - y ) * ( 2. - y ) / ( 1. - y ) - 8. * tf.sqrt(k2) * ( 2. - 2. * y + y * y ) * tf.cos( self.PI - (phi * self.RAD) ) )

  	# BH-DVCS interference squared amplitude eq (27) divided by e^6
        Amp2_I = 1. / ( x * y * y * y * t * P1 * P2 ) * ( A * ( F1 * ReH - t / 4. / self.M2 * F2 * ReE ) + B * ( F1 + F2 ) * ( ReH + ReE ) + C * ( F1 + F2 ) * ReHtilde )

        Amp2_I = self.GeV2nb * Amp2_I # convertion to nb

  	#return self.dsigma_I = Gamma1 * Amp2_I
        return Gamma1 * Amp2_I
        

    @tf.function
    def curve_fit_array(self, kins, cffs):
      # phi = tf.transpose(kins[:, 0])
      # kin1 = tf.transpose(kins[:, 1])
      # kin2 = tf.transpose(kins[:, 2])
      # kin3 = tf.transpose(kins[:, 3])
      # kin4 = tf.transpose(kins[:, 4])
      # F1 = tf.transpose(kins[:, 5])
      # F2 = tf.transpose(kins[:, 6])
      # ReH = tf.transpose(cffs[:, 0])
      # ReE = tf.transpose(cffs[:, 1])
      # ReHtilde = tf.transpose(cffs[:, 2])
      # c1fit = tf.transpose(cffs[:, 3])
      phi, QQ, x, t, k, F1, F2 = tf.split(kins, num_or_size_splits=7, axis=1)
      ReH, ReE, ReHtilde, c1fit = tf.split(cffs, num_or_size_splits=4, axis=1)

      ee, y, Gamma1, k2 = self.setKinematics(phi, QQ, x, t, k)
      P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, k, ee, y, k2)
      
      xsbhuu = self.BHUU(phi, QQ, x, t, k, F1, F2, ee, y, Gamma1, k2, P1, P2)
      xsiuu = self.IUU(phi, QQ, x, t, k, F1, F2, ReH, ReE, ReHtilde, y, Gamma1, k2, P1, P2)
      f_pred = xsbhuu + xsiuu + c1fit 
      return f_pred


    def curve_fit(self, kins, cffs):
      phi, QQ, x, t, k, F1, F2 = kins
      ReH, ReE, ReHtilde, c1fit = cffs
      
      ee, y, Gamma1, k2 = self.setKinematics(phi, QQ, x, t, k)
      P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, k, ee, y, k2)
      
      xsbhuu = self.BHUU(phi, QQ, x, t, k, F1, F2, ee, y, Gamma1, k2, P1, P2)
      xsiuu = self.IUU(phi, QQ, x, t, k, F1, F2, ReH, ReE, ReHtilde, y, Gamma1, k2, P1, P2)
      f_pred = xsbhuu + xsiuu + c1fit 
      return f_pred
