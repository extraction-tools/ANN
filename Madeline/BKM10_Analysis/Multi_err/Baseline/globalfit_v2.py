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

NUM_SETS = 500 ### 229, 1000

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
errors = [0.0, 0.1, 0.01, 0.05] #, 0.15, 0.2, 0.25, 0.025, 0.35, 0.5

i = int(sys.argv[1])
setI = data.getSet(i, itemsInSet=45)
results = pd.DataFrame({
      "ReH": np.zeros(NUM_SETS),
      "ReE": np.zeros(NUM_SETS),
      "ReHtilde": np.zeros(NUM_SETS),
      "SetNum": np.repeat(i, NUM_SETS)
    })

for error in errors:
    tfModel.set_weights(Wsave)
    
    for sample in range(NUM_SETS):
        #Number of kinematic sets
#         by_set = []
        tfModel.fit([data.Kinematics, data.XnoCFF], data.sampleY(error), # one replica of sampled F values across all sets
                            epochs=15000, verbose=0, batch_size=16, callbacks=[early_stopping_callback])
        cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)
        
        for num, cff in enumerate(['ReH', 'ReE', 'ReHtilde']):
            results.loc[sample, cff] = cffs[num]
            
#         by_set.append(cffs)
#     df = pd.DataFrame(by_set)
    results.to_csv('bySetCFFs' + sys.argv[1] + '_' + str(error) + '.csv')