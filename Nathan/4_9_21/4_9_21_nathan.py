# General method 4 that can be run with any dataset on Rivanna

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler

import utilities as uts

from BHDVCStf import BHDVCS #modified bhdvcs file

## System

import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/nbs9vy') # !!!! change to your computing ID !!!!

arg = int(sys.argv[1]) # feeds in through batch argument

## Setup

numSets = 15
numReplicas = 500

bhdvcs = BHDVCS()

df = pd.read_csv("dvcs_psuedo.csv") #change for specific dataset
data = uts.DvcsData(df)

localFits = np.fromfile('replicas500.txt')
localFits = localFits.reshape(numSets, numReplicas, 3)

## Define model

kinematics = tf.keras.Input(shape=(4))
x = tf.keras.layers.Dense(70, activation="elu")(kinematics)
x = tf.keras.layers.Dense(70, activation="elu")(x)
outputs = tf.keras.layers.Dense(3)(x)

globalModel = tf.keras.Model(inputs=kinematics, outputs=outputs, name="GlobalModel")

tf.keras.utils.plot_model(globalModel, "cffs.png", show_shapes=True)

X = data.Kinematics.loc[np.array(range(numSets))*36, :].reset_index(drop=True)

y = np.array(data.CFFs.loc[np.array(range(numSets))*36, :].reset_index(drop=True))

rescaler = MinMaxScaler()

rescaler = rescaler.fit(X)
X_rescaled = rescaler.transform(X)



# arg is in the range [0, 47] inclusive
opt_index = arg // 6
lr_index = arg % 6

# 8 different optimizers
optimizers = [
    tf.keras.optimizers.Adadelta, 
    tf.keras.optimizers.Adagrad, 
    tf.keras.optimizers.Adam, 
    tf.keras.optimizers.Adamax, 
    tf.keras.optimizers.Ftrl, 
    tf.keras.optimizers.Nadam, 
    tf.keras.optimizers.RMSprop, 
    tf.keras.optimizers.SGD
]

# 6 learning rates
learningRates = [
    0.001, # default for all of the optimizers
    0.005, 
    0.01, 
    0.05, 
    0.1, 
    0.5
]

globalModel.compile(optimizer=optimizers[opt_index](learning_rate=learningRates[lr_index]), loss=tf.keras.losses.MeanSquaredError())
orig_weights = globalModel.get_weights()

# Produce results

def produceResults(model, X, localFits, orig_weights, numSets, numReplicas, epochs=150):
    '''
    Essentially LOO cross-val with y-values being generated from seperate local fit
    
    globalModel: a tensorflow neural network model
    X: standardized kinematic variables
    orig_weights: the original weights from when the model was created (used to reset model after it has been trained)
    numSets: the number of kinematic sets
    numReplicas: the number of replicas
    
    returns: np array of cff predictions of shape (numSets, numReplicas, numCFFs)
    '''
    by_set = []
    for i in tqdm(range(numSets)):
        valid_x = X[[i], :]
        train_x = np.delete(X, i, axis=0)
        
        by_rep = []
        for rep in range(numReplicas):
            train_y = np.delete(localFits[:, rep, :], i, axis=0)
            
            model.set_weights(orig_weights)
            model.fit(train_x, train_y, epochs=epochs, verbose=0)
            by_rep.append(list(model.predict(valid_x)[0]))
        
        by_set.append(by_rep)

    return np.array(by_set)

results = produceResults(globalModel, X_rescaled, localFits, orig_weights, numSets, numReplicas)

# Prints for batch
pd.DataFrame(results[:, :, 0]).to_csv("/home/nbs9vy/Results_" + str(arg) + "_ReH.csv")
pd.DataFrame(results[:, :, 1]).to_csv("/home/nbs9vy/Results_" + str(arg) + "_ReE.csv")
pd.DataFrame(results[:, :, 2]).to_csv("/home/nbs9vy/Results_" + str(arg) + "_ReHtilde.csv")
