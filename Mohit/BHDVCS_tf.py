import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    def BHUU(self, phi, QQ, x, t, k, F1, F2, ee, y, Gamma1, k2, P1, P2):
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
    def IUU(self, phi, QQ, x, t, k, F1, F2, ReH, ReE, ReHtilde, y, Gamma1, k2, P1, P2):        
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

class DvcsData(object):
    def __init__(self, df):
      self.df = df
      self.X = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs']]
      self.XnoCFF = df.loc[:, ['phi_x', 'QQ', 'x_b', 't', 'k', 'F1', 'F2']]               
      self.y = df.loc[:, 'F']
      self.Kinematics = df.loc[:, ['k', 'QQ', 'x_b', 't']]
      self.erry = df.loc[:, 'errF']
        
    def __len__(self):
      return len(self.X)
    
    def getSet(self, setNum, itemsInSet=36):
      pd.options.mode.chained_assignment = None
      subX = self.X.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1, :]
      subX['F'] = self.y.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1]
      subX['errF'] = self.erry.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1]
      pd.options.mode.chained_assignment = 'warn'
      return DvcsData(subX)
    
    def sampleY(self):
      return np.random.normal(self.y, self.erry)
    
    def sampleWeights(self):
      return 1/self.erry
    
    def getAllKins(self, itemsInSets=36):
      return self.Kinematics.iloc[np.array(range(len(df)//itemsInSets))*itemsInSets, :]

def F2VsPhi(dataframe,SetNum,xdat,cffs,designation="overall"):
  f = BHDVCStf().curve_fit
  TempFvalSilces=dataframe[dataframe["#Set"]==SetNum]
  TempFvals=TempFvalSilces["F"]
  TempFvals_sigma=TempFvalSilces["errF"]
  temp_phi=TempFvalSilces["phi_x"]
  plt.errorbar(temp_phi,TempFvals,TempFvals_sigma,fmt='.',color='blue',label="Data")
  plt.xlim(0,368)
  temp_unit=(np.max(TempFvals)-np.min(TempFvals))/len(TempFvals)
  plt.ylim(np.min(TempFvals)-temp_unit,np.max(TempFvals)+temp_unit)
  plt.xticks(fontsize=15)
  plt.yticks(fontsize=15)
  plt.title("Local fit with data set #"+str(SetNum),fontsize=20)
  plt.plot(temp_phi, f(xdat,cffs), 'g--', label='fit')
  plt.xlabel("phi")
  plt.ylabel("F")
  plt.legend(loc=4,fontsize=10,handlelength=3)
  file_name = "plot_set_number_"+str(SetNum)+"_gridMetric_"+designation+".png"
  plt.savefig(file_name)
    
def cffs_from_globalModel(model, kinematics, numHL=1):
  '''
  :param model: the model from which the cffs should be predicted
  :param kinematics: the kinematics that should be used to predict
  :param numHL: the number of hidden layers:
  '''
  subModel = tf.keras.backend.function(model.layers[0].input, model.layers[numHL+2].output)
  return subModel(np.asarray(kinematics)[None, 0])[0]
    
def evaluate(y_yhat):
  '''
  Provides a few evaluation statistics from an array of true values and predictions.
  
  :param y_yhat: numpy array with first column being true values and second being predicted values.
  '''
  pct_errs = ((y_yhat[:, 0] - y_yhat[:, 1])/y_yhat[:, 1])*100
  print('Mean percent error: ', np.mean(np.abs(pct_errs)))

  RS = np.square((y_yhat[:, 0] - y_yhat[:, 1]))
  TS = np.square((y_yhat[:, 0] - y_yhat[:, 0].mean()))
  rmse = np.sqrt(np.mean(RS))
  rmtss = np.sqrt(np.mean(TS))
  print('RMSE: ', rmse)
  print('RMSE w yhat=mean: ', rmtss)
  RSS = np.sum(RS)
  TSS = np.sum(TS)
  print('R-squared: ', (1 - RSS/TSS))
  plt.hist(np.array(pct_errs))
  plt.title('Histogram of Percent Errors')
  plt.show()
    
def plotError(y_yhat, errs, var_string, title=None):
  '''
  :param y_yhat: numpy array of what it sounds like
  :param errs: list or array of stds of variable
  :param var_string: string of which variable is being plotted
  '''
  assert len(y_yhat) == len(errs)
  
  fig, ax = plt.subplots()
  ax.errorbar(x=list(range(len(errs))),
                y=list(map(lambda x: x[1], y_yhat)),
              yerr=errs,
              fmt='o',
              label="Estimated " + var_string)
  ax.plot(list(range(len(errs))),
            list(map(lambda x: x[0], y_yhat)),
            'ro', label="Actual " + var_string)
  
  ax.set_xlabel("Set#")
  ax.set_ylabel(var_string)
  if title != None:
      ax.set_title(title)
      
  ax.legend()
  plt.show()
