#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
# from BHDVCS_tf import BHDVCStf
from BHDVCS_tf import TotalFLayer
from BHDVCS_tf import DvcsData
from BHDVCS_tf import cffs_from_globalModel
from BHDVCS_tf import F2VsPhi as F2VsPhitf
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

import matplotlib
import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare
from tqdm import tqdm


# In[2]:


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


# In[ ]:


initializer = tf.keras.initializers.HeNormal()

kinematics = tf.keras.Input(shape=(7))
x1 = tf.keras.layers.Dense(200, activation="relu", kernel_initializer=initializer)(kinematics)
x2 = tf.keras.layers.Dense(200, activation="relu", kernel_initializer=initializer)(x1)
outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x2)
# noncffInputs = tf.keras.Input(shape=(7))
# #### phi, kin1, kin2, kin3, kin4, F1, F2 ####
# total_FInputs = tf.keras.layers.concatenate([noncffInputs,outputs])
TotalF = TotalFLayer()(outputs)

tfModel = tf.keras.Model(inputs=kinematics, outputs = TotalF, name="tfmodel")
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


# In[6]:




#!!High-overfitting from batch_size 1, 2 100 node hidden layers no validation data, huge number of epochs!!#
# Over-fitting to F will likely not reflect well in CFF predictions

#Number of kinematic sets
by_set = []
number = 0
for i in tqdm(range(Total_Sets)):
    
  setI = data.getSet(i,start_index,N_data)

#   Data = pd.concat([setI.Kinematics, setI.XnoCFF], axis=1)

#   Data_v2 = pd.concat([Data, pd.DataFrame(setI.sampleY())], axis=1)

#   Data_v3 = Data_v2.rename(columns={Data_v2.columns[11]: 'F'})

#   y = Data_v3['F']
#   x = Data_v3.drop(['F'],axis=1)
    
#   train_X, test_X, train_Y, test_Y = train_test_split(x, y, test_size=0.2)


  tfModel.set_weights(Wsave)

#   hist = tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(), # one replica of samples from F vals
                        
#   epochs=150, verbose=0, batch_size=16)

#   hist = tfModel.fit(train_X,train_Y, validation_data = (test_X, test_Y),  # one replica of samples from F vals
                        
#   epochs=150, verbose=0, batch_size=16)

  hist = tfModel.fit(setI.XnoCFF, setI.sampleY(), # one replica of samples from F vals
                        
  epochs=50, verbose=0, batch_size=2056)
  
  cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

  by_set.append(cffs)
  
df = pd.DataFrame(by_set)
df.to_csv(f"bySetCFFs(195).csv")

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val_loss'], loc='upper left')
plt.savefig('Loss_Plot_V3_tuned.png')
# df.to_csv(f"/project/ptgroup/Devin/ANN/BKM_T/CFF_Data/bySetCFFs(195)_" + sys.argv[1] + ".csv")


# In[6]:


# setI = data.getSet(0,start_index,N_data)

# Data = pd.concat([setI.Kinematics, setI.XnoCFF], axis=1)

# Data_v2 = pd.concat([Data, pd.DataFrame(setI.sampleY())], axis=1)

# Data_v3 = Data_v2.rename(columns={Data_v2.columns[11]: 'F'})

# y = Data_v3['F']
# x = Data_v3.drop(['F'],axis=1)

# # print(y)
# # trainX, testX, trainY, testY = trn_tst(x,y)

# # trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.1)

# print(Data_v3)


# In[34]:


# testY


# In[ ]:




