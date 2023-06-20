import numpy as np
import pandas as pd 
from BHDVCS_tf_v2 import TotalFLayer
from BHDVCS_tf_v2 import DvcsData
from BHDVCS_tf_v2 import cffs_from_globalModel
from BHDVCS_tf_v2 import F2VsPhi as F2VsPhitf
import tensorflow as tf
import sys

import matplotlib
import matplotlib.pyplot as plt

df = pd.read_csv("pseudoKM15_New_FormFactor.csv", dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

N_data = []
start_index = []
n_total = 0
Total_Sets = 195
for i in range(Total_Sets):
    TempFvalSilces=df[df["#Set"]==i+1]
    TempFvals=TempFvalSilces["F"]
    start_index = np.append(start_index, n_total)
    N_data = np.append(N_data, TempFvals.size - 1)
    n_total = n_total + TempFvals.size

data = DvcsData(df)

initializer = tf.keras.initializers.HeNormal()

kinematics = tf.keras.Input(shape=(4))
x = kinematics

num_hidden_layers = 20
hidden_units = 100
activation = "tanh"

for _ in range(num_hidden_layers):
    x = tf.keras.layers.Dense(hidden_units, activation=activation, kernel_initializer=initializer)(x)

outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x)

noncffInputs = tf.keras.Input(shape=(7))
total_FInputs = tf.keras.layers.concatenate([noncffInputs, outputs])
TotalF = TotalFLayer()(total_FInputs)

tfModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs=TotalF, name="tfmodel")
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000005, patience=100)

lr = tf.keras.optimizers.schedules.ExponentialDecay(
    0.0085, df.shape[0]/1, 0.96, staircase=False, name=None
)

tfModel.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss=tf.keras.losses.MeanSquaredError(),
    run_eagerly=True
)

Wsave = tfModel.get_weights()

by_set = []

for i in range(Total_Sets):
    setI = data.getSet(i, start_index, N_data)
    tfModel.set_weights(Wsave)

    tfModel.fit(
        [setI.Kinematics, setI.XnoCFF], setI.sampleY(),
        epochs=500, verbose=0, batch_size=2048
    )
  
    cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=num_hidden_layers)
    by_set.append(cffs)
  
df = pd.DataFrame(by_set)
df.to_csv(f"./CFF_Data/bySetCFFs(195)_" + sys.argv[1] + ".csv")
