import numpy as np
import pandas as pd 
# from BHDVCS_tf import BHDVCStf
from BHDVCS_tf import TotalFLayer
from BHDVCS_tf import DvcsData
from BHDVCS_tf import cffs_from_globalModel
from BHDVCS_tf import F2VsPhi as F2VsPhitf
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

df = pd.read_csv("BKM_pseudodata.csv", dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

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
    loss = tf.keras.losses.MeanSquaredError()
)

Wsave = tfModel.get_weights()

#!!High-overfitting from batch_size 1, 2 100 node hidden layers no validation data, huge number of epochs!!#
# Over-fitting to F will likely not reflect well in CFF predictions

#Number of kinematic sets
by_set = []
for i in range(194):
  setI = data.getSet(i, itemsInSet=45)

  tfModel.set_weights(Wsave)

  tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(), # one replica of samples from F vals
                        epochs=15000, verbose=0, batch_size=16, callbacks=[early_stopping_callback])
  
  cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

  by_set.append(cffs)

df = pd.DataFrame(by_set)

if len(sys.argv) > 1:
    df.to_csv('bySetCFFs' + sys.argv[1] + '.csv')
else:
    df.to_csv('bySetCFFs.csv')
