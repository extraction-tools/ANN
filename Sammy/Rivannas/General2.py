#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from datetime import datetime
import sys

print(tf.__version__)

print("\n \n I'm running! \n \n")
# In[2]:

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/sl8rn/Rivannas')

from BHDVCStf import BHDVCS #modified bhdvcs file
bhdvcs = BHDVCS()


# In[37]:


df = pd.read_csv('/home/sl8rn/Rivannas/dvcs_psuedo.csv')

#Setting the hyperparameters here:
learningRate = random.uniform(0.1,0.0000001)
activationFunctions = ["tanh", "relu", "selu", "elu"]
activationSeed = random.randint(0,3)
activationFunction = activationFunctions[activationSeed]

possibleNodes = [4,8,20,16,32,64,128]
nodeSeed = random.randint(0,6)
#numNodes = 4
numNodes = possibleNodes[nodeSeed]
#numNodes = 20
numLayers = 1
#Number of layers: 1,2,3,4
#Number of nodes
#Optimizers:
#Explore dropout layers upperbound: 1-2(non-dropout: 2-4 layers), and dropout rate

# In[4]:


class DvcsData(object):
    def __init__(self, df):
        self.X = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'ReH', 'ReE', 'ReHtilde', 'dvcs']]
        self.XnoCFF = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs']]
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


# In[5]:


data = DvcsData(df)


# In[6]:


class TotalUUXS(tf.keras.layers.Layer):
    def __init__(self):
        super(TotalUUXS, self).__init__(dtype='float64')
        self.F = BHDVCS()
    def call(self, inputs):
        return self.F.TotalUUXS(inputs[:, :8], inputs[:, 8], inputs[:, 9], inputs[:, 10])


# In[7]:

#Number of layers, number of nodes, optimizers 
#Number of nodes hyperparameter
kinematics = tf.keras.Input(shape=(4))
x = tf.keras.layers.Dense(numNodes, activation=activationFunction)(kinematics)
outputs = tf.keras.layers.Dense(3)(x) #three output nodes for ReH, ReE, ReHtilde

noncffInputs = tf.keras.Input(shape=(8))
totalUUXSInputs = tf.keras.layers.concatenate([noncffInputs, outputs])
F = TotalUUXS()(totalUUXSInputs)

globalModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs=F, name="GlobalModel")


# In[8]:


globalModel.compile(
    optimizer = tf.keras.optimizers.Adam(learningRate),
    loss = tf.keras.losses.MeanSquaredError(),
)




# In[11]:


def cffs_from_globalModel(model, kinematics):
    subModel = tf.keras.backend.function(model.layers[0].input, model.layers[3].output)
    return subModel(np.asarray(kinematics)[None, 0])[0]


# In[16]:


numSamples = 200

i = int(sys.argv[1])


setI = data.getSet(i)

results = pd.DataFrame({
      "ReH": np.zeros(numSamples),
      "ReE": np.zeros(numSamples),
      "ReHtilde": np.zeros(numSamples),
      "SetNum": np.repeat(i, numSamples)
    })


for sample in range(numSamples):

        chkpt_path = 'best-network' + str(i) + '.hdf5'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=chkpt_path,
            save_weights_only=True,
            monitor='loss',
            mode='min',
            save_best_only=True)

        globalModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(),
                    epochs=2500, verbose=0,
                    callbacks=[model_checkpoint_callback])

        globalModel.load_weights(chkpt_path)

        cffs = cffs_from_globalModel(globalModel, setI.Kinematics)

        for num, cff in enumerate(['ReH', 'ReE', 'ReHtilde']):
            results.loc[sample, cff] = cffs[num]      
      
# In[86]:


results.to_csv("/home/sl8rn/Rivannas/Results/Results"+ str(i) + ".csv")
f = open("/home/sl8rn/Rivannas/Results/modelResults" + str(i) + ".txt", "x")
f.write("Learning Rate: " + str(learningRate) + "\n")
f.write("ActivationFunction: " + activationFunction + "\n")
f.write("Number of Nodes: " + str(numNodes) + "\n")
f.write("Number of Hidden Layers: " + str(numLayers) + "\n")
f.close()
