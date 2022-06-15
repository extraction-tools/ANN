import numpy as np
import pandas as pd 
# from BHDVCS_tf import BHDVCStf
# from BHDVCS_tf import TotalFLayer
# from BHDVCS_tf import DvcsData
# from BHDVCS_tf import cffs_from_globalModel
# from BHDVCS_tf import F2VsPhi
from BHDVCS_tf import *
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

df = pd.read_csv("BKM_pseudodata.csv", dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

data = DvcsData(df)

kinematics = tf.keras.Input(shape=(4))
x1 = tf.keras.layers.Dense(100, activation="tanh")(kinematics)
x2 = tf.keras.layers.Dense(100, activation="tanh")(x1)
outputs = tf.keras.layers.Dense(4, activation="linear")(x2)
noncffInputs = tf.keras.Input(shape=(7))
#### phi, kin1, kin2, kin3, kin4, F1, F2 ####
total_FInputs = tf.keras.layers.concatenate([noncffInputs,outputs])
TotalF = TotalFLayer()(total_FInputs)

tfModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs = TotalF, name="tfmodel")
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000005, patience=25)

tfModel.compile(
    optimizer = tf.keras.optimizers.Adam(.0085),
    loss = tf.keras.losses.MeanSquaredError()
)

Wsave = tfModel.get_weights()

#!!High-overfitting from batch_size 1, 2 100 node hidden layers no validation data, huge number of epochs!!#
# Over-fitting to F will likely not reflect well in CFF predictions

def get_total_error(experimental_values, expected_values):
  experimental_values, expected_values = list(experimental_values), list(expected_values)
  tot = 0
  for i,j in zip(experimental_values, expected_values):
    tot += abs(float(j) - float(i))
  return tot

def get_max_residual(x_values, experimental_values, expected_values):
  x_values, experimental_values, expected_values = list(x_values), list(experimental_values), list(expected_values)
  max = 0
  maxIndex = 0
  for n, (i,j) in enumerate(zip(experimental_values, expected_values)):
    residual = abs(float(j) - float(i))
    if residual > max:
      max = residual
      maxIndex = n
  return (x_values[n], max)

def F2VsPhi_noPlot(dataframe,SetNum,xdat,cffs):
  f = BHDVCStf().curve_fit
  TempFvalSilces=dataframe[dataframe["#Set"]==SetNum]
  TempFvals=TempFvalSilces["F"]
  TempFvals_sigma=TempFvalSilces["errF"]
  temp_phi=TempFvalSilces["phi_x"]

  calculated_points = f(xdat,cffs)

  return get_total_error(calculated_points, TempFvals), get_max_residual(temp_phi, calculated_points, TempFvals)


################################################# FINDING THE BEST COMBINATION OF EPOCH AND BATCH #######################################
total_errors = {}
total_residuals = {}

best_combination_errors = {0:(0,0,100), 1:(0,0,100), 2:(0,0,100), 3:(0,0,100), 4:(0,0,100)} #best errors for each set

best_combination_residual = {0:(0,0,100), 1:(0,0,100), 2:(0,0,100), 3:(0,0,100), 4:(0,0,100)} #best residuals for each set

for epoch in np.arange(5,1001,10):
  for batch in np.arange(1,10,1):
    by_set = []
    for i in range(5):
      setI = data.getSet(i, itemsInSet=45)

      tfModel.set_weights(Wsave)

      tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(), # one replica of samples from F vals
                            epochs=epoch, verbose=0, batch_size=batch, callbacks=[early_stopping_callback])
      
      
      cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

      by_set.append(cffs)

      new_xdat = np.transpose(setI.XnoCFF.to_numpy(dtype=np.float32)) #NB: Could rewrite BHDVCS curve_fit to not require transposition

      # Avoid recalculating F-values from cffs when that is what the model is predicting already
      total_error, max_residual = F2VsPhi_noPlot(df,i+1,new_xdat,cffs); #runs the version without plotting to save time
      total_errors[(epoch, batch, i)] = total_error
      total_residuals[(epoch, batch, i)] = max_residual

      if best_combination_errors[i][2] > total_error:
        best_combination_errors[i] = (epoch, batch, total_error)

      if best_combination_residual[i][2] > max_residual[1]:
        best_combination_residual[i] = (epoch, batch, max_residual[1])

most_common = []
for i,j in zip(best_combination_errors.values(), best_combination_residual.values()):
  most_common.append(i[:2])
  most_common.append(j[:2])

final_outcome = max(set(best_combination_residual.values()), key = list(best_combination_residual.values()).count)
print("Just be residuals, the best epoch number is:", final_outcome[0], "with a batch size of", final_outcome[1])

final_outcome = max(set(best_combination_errors.values()), key = list(best_combination_errors.values()).count)
print("Just be residuals, the best epoch number is:", final_outcome[0], "with a batch size of", final_outcome[1])

final_outcome = max(set(most_common), key=most_common.count) #the final_outcome is a tuple of (epoch#, batch#)
print("The best epoch number is: ", final_outcome[0], "with a batch size of", final_outcome[1])


by_set = []
for i in range(5): #use the final outcome to have a final fit
  setI = data.getSet(i, itemsInSet=45)

  tfModel.set_weights(Wsave)

  tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(), # one replica of samples from F vals
                        epochs=final_outcome[0], verbose=0, batch_size=final_outcome[1], callbacks=[early_stopping_callback])
  
  
  cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

  by_set.append(cffs)

  new_xdat = np.transpose(setI.XnoCFF.to_numpy(dtype=np.float32)) #NB: Could rewrite BHDVCS curve_fit to not require transposition

  # Avoid recalculating F-values from cffs when that is what the model is predicting already
  F2VsPhi(df,i+1,new_xdat,cffs);
  plt.clf()


df = pd.DataFrame(by_set)
df.to_csv('bySetCFFs.csv')
