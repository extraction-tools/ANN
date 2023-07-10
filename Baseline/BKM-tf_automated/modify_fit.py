import nbformat as nbf
import pandas as pd
import os

#note that the /content/foldercff must be replaced with where you want the folder with the cff to go. It then needs to be copied into the os.chir
#also /content/Book 8.csv is the location of the csv file that reads the data.
#later the /content/PseudoKM15_New_FormFactor.csv is the file for the training data.

os.makedirs(r"/content/foldercff")
nb = nbf.v4.new_notebook()
ddf=pd.read_csv(r"/content/BKM_results.csv",header=0,
                usecols=["# of layers", "nodes first layer", "decreasing nodes",
                         "activation function","initial learning rate","decay rate"])
os.chdir(r"/content/foldercff")
#note to call funcitons use ddf.iloc[row][column] with normal python index at 0
#to have varibles must use {} around the varibels.
#note that if the code has {} in it must put {} around it to make work ie {simga:sigma} must be {{simga:sigma}}

for j in range(len(ddf.iloc[:,0])):
    code = f"""import numpy as np
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

df = pd.read_csv(r"/content/PseudoKM15_New_FormFactor.csv", dtype=np.float64)
df = df.rename(columns={{"sigmaF": "errF"}})

data = DvcsData(df)

initializer = tf.keras.initializers.HeNormal()

kinematics = tf.keras.Input(shape=(4))

k=0
while k <{ddf.iloc[j,0]}:
  if k==0:
      x0=tf.keras.layers.Dense({ddf.iloc[j][1]}, activation="{ddf.iloc[j][3]}", kernel_initializer=initializer)(kinematics)
  else:
      xo=x0
      x0=tf.keras.layers.Dense({ddf.iloc[j][1]}-k*{ddf.iloc[j][2]}, activation="{ddf.iloc[j][3]}", kernel_initializer=initializer)(xo)
  k=k+1
outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x0)
noncffInputs = tf.keras.Input(shape=(7))
#### phi, kin1, kin2, kin3, kin4, F1, F2 ####
total_FInputs = tf.keras.layers.concatenate([noncffInputs,outputs])
TotalF = TotalFLayer()(total_FInputs)

tfModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs = TotalF, name="tfmodel")
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000005, patience=100)

lr = tf.keras.optimizers.schedules.ExponentialDecay(
    {ddf.iloc[j][4]}, df.shape[0]/1, {ddf.iloc[j][5]}, staircase=False, name=None
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
for i in range(15):
  setI = data.getSet(i, itemsInSet=45)

  tfModel.set_weights(Wsave)

  tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(), # one replica of samples from F vals
                        epochs=1001, verbose=0, batch_size=16, callbacks=[early_stopping_callback])

  cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

  by_set.append(cffs)

df = pd.DataFrame(by_set)

if len(sys.argv) > 1:
    df.to_csv('bySetCFFs' + sys.argv[1] + str({j})+'.csv')
  else:
    df.to_csv('bySetCFFs' + str({j}) +'.csv')
    """
    nb['cells'] = [nbf.v4.new_code_cell(code)]
    fname = 'test'+str(j)+'.ipynb'
    with open(fname, 'w') as f:
      nbf.write(nb, f)
