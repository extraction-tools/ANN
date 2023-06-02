import numpy as np
import pandas as pd 
# from BHDVCS_tf import BHDVCStf
from BHDVCS_tf import TotalFLayer
from BHDVCS_tf import DvcsData
from BHDVCS_tf import cffs_from_globalModel
from BHDVCS_tf import F2VsPhi as F2VsPhitf
import tensorflow as tf
import sys

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

df = pd.read_csv("PseudoKM15_New_FormFactor.csv", dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

N_data = []
start_index = []
n_total = 0
Total_Sets = 195
for i in range(Total_Sets):
  TempFvalSilces=df[df["#Set"]==i+1]
  TempFvals=TempFvalSilces["F"]
  start_index = np.append(start_index, n_total)
  N_data = np.append(N_data, TempFvals.size)
  n_total = n_total + TempFvals.size

data = DvcsData(df)

initializer = tf.keras.initializers.HeNormal()

kinematics = tf.keras.Input(shape=(4))
x1 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(kinematics)
x2 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(x1)
outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x2)
noncffInputs = tf.keras.Input(shape=(7))
#### phi, kin1, kin2, kin3, kin4, F1, F2 ####
total_FInputs = tf.keras.layers.concatenate([noncffInputs,outputs])
TotalF = TotalFLayer()(total_FInputs)

tfModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs = TotalF, name="tfmodel")
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000005, patience=100)

lr = tf.keras.optimizers.schedules.ExponentialDecay(
    0.0085, df.shape[0]/1, 0.96, staircase=False, name=None
)


tfModel.compile(
    optimizer = tf.keras.optimizers.Adam(lr),
    loss = tf.keras.losses.MeanSquaredError(),
    run_eagerly=True
)

Wsave = tfModel.get_weights()

#!!High-overfitting from batch_size 1, 2 100 node hidden layers no validation data, huge number of epochs!!#
# Over-fitting to F will likely not reflect well in CFF predictions

#Number of kinematic sets
by_set = []
number = 0
for i in range(Total_Sets):
  setI = data.getSet(i,start_index,N_data)

  tfModel.set_weights(Wsave)

  tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(), # one replica of samples from F vals
                        
  epochs=15000, verbose=0, batch_size=16, callbacks=[early_stopping_callback])
  
  cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

  by_set.append(cffs)
  
df = pd.DataFrame(by_set)
df.to_csv(f"/project/ptgroup/Devin/ANN/BKM_T/CFF_Data/bySetCFFs(195)_" + sys.argv[1] + ".csv")

