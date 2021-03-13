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
sys.path.insert(1, '/home/ncn6mq') # !!!! change to your computing ID !!!!

i = int(sys.argv[1]) # feeds in through batch argument

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
x = tf.keras.layers.Dense(20, activation="elu")(kinematics)
x = tf.keras.layers.Dense(20, activation="elu")(x)
outputs = tf.keras.layers.Dense(3)(x)

globalModel = tf.keras.Model(inputs=kinematics, outputs=outputs, name="GlobalModel")

tf.keras.utils.plot_model(globalModel, "cffs.png", show_shapes=True)

X = data.Kinematics.loc[np.array(range(numSets))*36, :].reset_index(drop=True)

y = np.array(data.CFFs.loc[np.array(range(numSets))*36, :].reset_index(drop=True))

rescaler = MinMaxScaler()

rescaler = rescaler.fit(X)
X_rescaled = rescaler.transform(X)

globalModel.compile(optimizer=tf.keras.optimizers.Adam(.1), loss=tf.keras.losses.MeanSquaredError())
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

results.to_csv("/home/ncn6mq/Results"+ str(i) + ".csv") #prints for batch