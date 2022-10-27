#!/usr/bin/env python
# coding: utf-8

# In[1]:

#MAKE SURE FORMULATION IS CORRECT: AROUND LINE 28
#MAKE SURE SIGMAF V ERRF IS DEFINED IN THE DVCS OBJECT CLASS BASED ON DATASET USED


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from datetime import datetime
# from BHDVCS_tf import TotalFLayer
# from BHDVCS_tf import DvcsData
# from BHDVCS_tf import cffs_from_globalModel
import sys

print(tf.__version__)

print("\n \n I'm running! OG Status \n \n")
# In[2]:

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/mag4ka/Aaryan/Rivanna/Multi_err')
# python ErrorFAnalysis_Method2.py /home/mag4ka/Analysis


# from BHDVCS_tf import BHDVCStf
# bhdvcs = BHDVCStf()

from TVA1_UU import TVA1_UU #modified bhdvcs file
bhdvcs = TVA1_UU()


# In[37]:


df = pd.read_csv('/home/mag4ka/Aaryan/dvcs_bkm_xs_June2021_4pars.csv')


# In[4]:
data = DvcsData(df) 

# In[7]:
kinematics = tf.keras.Input(shape=(4))
x = tf.keras.layers.Dense(20, activation="tanh")(kinematics)
outputs = tf.keras.layers.Dense(4)(x) #three output nodes for ReH, ReE, ReHtilde, c1fit
noncffInputs = tf.keras.Input(shape=(7))
total_FInputs = tf.keras.layers.concatenate([noncffInputs, outputs])
F = TotalFLayer()(total_FInputs)

globalModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs=F, name="GlobalModel") 


# In[8]:
globalModel.compile(
    optimizer = tf.keras.optimizers.Adam(0.02),
    loss = tf.keras.losses.MeanSquaredError(),
)

# In[11]:
numSamples = 1000
originalWeights = globalModel.get_weights()
## TESTING
# print(sys.argv[1])
# print(type(sys.argv[1]))
# python ErrorFAnalysis_Method2.py /tmp
i = int(sys.argv[1])
errors = [0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.30]

setI = data.getSet(i)

results = pd.DataFrame({
      "ReH": np.zeros(numSamples),
      "ReE": np.zeros(numSamples),
      "ReHtilde": np.zeros(numSamples),
      "SetNum": np.repeat(i, numSamples)
    })

#Using unrelated bootstrapping method
#globalModel.load_weights('/home/knc8xp/method2Weights.h5')
for error in errors:
    globalModel.set_weights(originalWeights)
    for sample in range(numSamples):
        globalModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(error),
                    epochs=2500, verbose=0)
        
        cffs = cffs_from_globalModel(globalModel, setI.Kinematics)
        
        for num, cff in enumerate(['ReH', 'ReE', 'ReHtilde']):
            results.loc[sample, cff] = cffs[num]      
      
    results.to_csv("/home/mag4ka/Aaryan/Predictions/Method2-" + str(i) + "-" + str(error) + ".csv")