import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler

import utilities as uts

from BHDVCStf import BHDVCS #modified bhdvcs file


numSets = 15
numReplicas = 100

df = pd.read_csv("dvcs_xs_newsets_genCFFs.csv")
data = uts.DvcsData(df)

## Define model

kinematics = tf.keras.Input(shape=(4))
x = tf.keras.layers.Dense(20, activation="elu")(kinematics)
outputs = tf.keras.layers.Dense(3)(x) #three output nodes for ReH, ReE, ReHtilde
noncffInputs = tf.keras.Input(shape=(8))
totalUUXSInputs = tf.keras.layers.concatenate([noncffInputs, outputs])
F = uts.TotalUUXSlayer()(totalUUXSInputs) # incorporate cross-sectional function

globalModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs=F, name="GlobalModel")


globalModel.compile(
    optimizer = tf.keras.optimizers.Adam(.01),
    loss = tf.keras.losses.MeanSquaredError(),
)

orig_weights = globalModel.get_weights()

## Data

rescaler = MinMaxScaler() # need to scale data for neural network

X = data.Kinematics.loc[np.array(range(numSets))*36, :]
rescaler = rescaler.fit(data.Kinematics)
X_rescaled = rescaler.transform(X)

xnocff = np.array(data.XnoCFF.loc[np.array(range(numSets))*36, :])

## Fit

def produceResults(model, X, xnocff, data, orig_weights, numSets, numReplicas, epochs=100, whichPhi=18):
    '''
    Essentially LOO cross-val with y-values being generated from seperate local fit
    
    globalModel: a tensorflow neural network model
    X: [standardized kins, xnocff]
    orig_weights: the original weights from when the model was created (used to reset model after it has been trained)
    numSets: the number of kinematic sets
    numReplicas: the number of replicas
    
    returns: np array of cff predictions of shape (numSets, numReplicas, numCFFs)
    '''
    by_set = []
    for i in tqdm(range(numSets)):
        valid_x = X[[i], :] # specific row of standardized kins
        
        train_x = np.delete(X, i, axis=0)
        train_xnocff = np.delete(xnocff, i, axis=0)
        
        msk = np.delete(np.array(range(numSets))*36 + whichPhi, i) # grab only Fs for phi=whichPhi
        
        by_rep = []
        for rep in range(numReplicas):
            train_y = np.array(data.sampleY()[msk])
            
            model.set_weights(orig_weights)
            
            #return model, [train_x, train_xnocff], train_y, valid_x
            model.fit([train_x, train_xnocff], train_y, epochs=epochs, verbose=0)
            by_rep.append(list(uts.cffs_from_globalModel(model, valid_x)))
                    
        by_set.append(by_rep)

    return np.array(by_set)

results = produceResults(globalModel, X_rescaled, xnocff, data, orig_weights, numSets, 10)

i = int(sys.argv[1])

results.to_csv("/home/ncn6mq/Results"+ str(i) + ".csv")
