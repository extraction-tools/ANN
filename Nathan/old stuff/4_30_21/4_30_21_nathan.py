# Method 6 hyperparameter search
# Nathan


import pandas as pd
import numpy as np
import scipy.optimize as optimization
import utilities as uts
from BHDVCStf import BHDVCS
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

from GlobalFittingSchema import GlobalFittingSchema

import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/ecp5kf/ANN/nathan_jobs/')

arg = int(sys.argv[1])

g = GlobalFittingSchema()

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

optimizerNames = [
    'Adadelta',
    'Adagrad',
    'Adam',
    'Adamax',
    'Ftrl',
    'Nadam',
    'RMSprop',
    'SGD'
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

runName : str = optimizerNames[opt_index] + '-' + str(learningRates[lr_index]) + '-Nathan'

g.addArchitecture('Method6-2DenseLayers')
g.addRun('Method6-2DenseLayers', runName)
g.setHyperparameter('Method6-2DenseLayers', runName, 'optimizer', optimizerNames[opt_index])
g.setHyperparameter('Method6-2DenseLayers', runName, 'learning-rate', learningRates[lr_index])





bhdvcs = BHDVCS()
df = pd.read_csv("dvcs_xs_newsets_genCFFs.csv")
data = uts.DvcsData(df)
filename="dvcs_xs_newsets_genCFFs.csv"

numSets = 15
numReplicas = 10



def Histogram_Data(fitname,setnum,cffnum):
    Cffdat=[]
    LL=len(fitname[setnum])
    for i in range(LL):
        Cffdat.append(fitname[setnum][i][cffnum])
    return np.array(Cffdat)

fit_open = np.fromfile("replicas500.txt")
fit_open = fit_open.astype(np.float32)

reshaped_data = fit_open.reshape(15, 500, 3)
reshaped_data = reshaped_data.astype(np.float32)

localFits = fit_open.reshape(15, 500, 3)
localFits = localFits.astype(np.float32)


"""## Defining NN A"""

CFF_replicas=reshaped_data[:, :36, :].reshape(15*36, 3)
CFF_replicas = CFF_replicas.astype(np.float32)



"""## Data and Imports"""

X1 = data.Kinematics
X1 = X1.astype(np.float32)

X2 = data.XnoCFF
X2 = X1.astype(np.float32)


y1 = localFits
y1 = y1.astype(np.float32)

y2 = data.y
y2 = y2.astype(np.float32)

rescaler = MinMaxScaler()

rescaler = rescaler.fit(X1)
X1 = rescaler.transform(X1)
X1 = X1.astype(np.float32)


"""# Model definition"""

kins = tf.keras.Input(shape=(4), dtype=tf.float32)
x = tf.keras.layers.Dense(20, activation="relu")(kins)
x = tf.keras.layers.Dense(20, activation="relu")(x)
cffs = tf.keras.layers.Dense(3)(x) #three output nodes for ReH, ReE, ReHtilde
noncffInputs = tf.keras.Input(shape=(8), dtype=tf.float32)
totalUUXSInputs = tf.keras.layers.concatenate([noncffInputs, cffs])
F = uts.TotalUUXSlayer()(totalUUXSInputs) # incorporate cross-sectional function

globalModel = tf.keras.Model(inputs=[kins, noncffInputs], outputs=[cffs, F], name="GlobalModel")

orig_weights = globalModel.get_weights()

tf.keras.utils.plot_model(globalModel, "cffs.png", show_shapes=True)

globalModel.compile(
    optimizer = tf.keras.optimizers.Adam(lr=0.005),
    loss = [tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError()],
)


def train_test(arr, obsinset, whichset):
    train = np.delete(arr, slice((obsinset*whichset), (obsinset*(whichset+1))), axis=0)
    test = arr[(obsinset*whichset):(obsinset*(whichset+1)), :]
    return train, test

def produceResults(model, data, localFits, orig_weights, numSets, numReplicas, epochs=100):
    '''
    Essentially LOO cross-val with y-values being generated from seperate local fit

    globalModel: a tensorflow neural network model
    kins: rescaled kinematics
    X: dvcsdata of whole set
    localFits: the
    orig_weights: the original weights from when the model was created (used to reset model after it has been trained)
    numSets: the number of kinematic sets
    numReplicas: the number of replicas

    returns: np array of cff predictions of shape (numSets, numReplicas, numCFFs)
    '''

    X1=np.array(data.Kinematics)
    X2=np.array(data.XnoCFF)

    #y1=localFits[:, :36, :].reshape(numSets*36,3)
    y2=np.array(data.y).reshape(-1, 1)

    rescaler = MinMaxScaler()

    rescaler = rescaler.fit(X1)
    X1 = rescaler.transform(X1)

    by_set = []
    for i in range(numSets):

        train_X1, valid_X1 = train_test(X1, 36, i)
        train_X2, valid_X2 = train_test(X2, 36, i)

        by_rep = []
        for rep in range(numReplicas):

            idxs = np.random.choice(list(range(localFits.shape[1])), size=36, replace=False)
            y1 = localFits[:, idxs, :].reshape(numSets*36, 3)
            train_y1, test_y1 = train_test(y1, 36, i)

            train_weights, _ = train_test(1/np.array(data.erry).reshape(-1, 1), 36, i)

            y2 = np.array(data.sampleY()).reshape(-1, 1)
            train_y2, test_y2 = train_test(y2, 36, i)
            train_y2 = train_y2[:, 0]

            model.set_weights(orig_weights)
            model.fit(x=[train_X1, train_X2], y=[train_y1, train_y2], sample_weight={'total_uux_slayer': train_weights},
                      epochs=epochs, verbose=0)

            by_rep.append(list(model.predict([valid_X1, valid_X2])[0][0, :]))

        by_set.append(by_rep)
    return np.array(by_set), rescaler

results, rescaler = produceResults(globalModel, data, localFits, orig_weights, 15, 20)



"""# ReH"""

y_yhat, err = uts.y_yhat_errCFFs(data, results, 0)

pct_errs = ((y_yhat[:, 0] - y_yhat[:, 1])/y_yhat[:, 1])*100
g.setStatistic('Method6-2DenseLayers', runName, 'ReH', 'MPE', np.mean(np.abs(pct_errs)))

RS = np.square((y_yhat[:, 0] - y_yhat[:, 1]))
TS = np.square((y_yhat[:, 0] - y_yhat[:, 0].mean()))
rmse = np.sqrt(np.mean(RS))
rmtss = np.sqrt(np.mean(TS))
g.setStatistic('Method6-2DenseLayers', runName, 'ReH', 'RMSE', rmse)
g.setStatistic('Method6-2DenseLayers', runName, 'ReH', 'RMTSS', rmtss)

RSS = np.sum(RS)
TSS = np.sum(TS)
g.setStatistic('Method6-2DenseLayers', runName, 'ReH', 'R^2', (1 - RSS/TSS))



"""# ReE"""

y_yhat, err = uts.y_yhat_errCFFs(data, results, 1)


pct_errs = ((y_yhat[:, 0] - y_yhat[:, 1])/y_yhat[:, 1])*100
g.setStatistic('Method6-2DenseLayers', runName, 'ReE', 'MPE', np.mean(np.abs(pct_errs)))

RS = np.square((y_yhat[:, 0] - y_yhat[:, 1]))
TS = np.square((y_yhat[:, 0] - y_yhat[:, 0].mean()))
rmse = np.sqrt(np.mean(RS))
rmtss = np.sqrt(np.mean(TS))
g.setStatistic('Method6-2DenseLayers', runName, 'ReE', 'RMSE', rmse)
g.setStatistic('Method6-2DenseLayers', runName, 'ReE', 'RMTSS', rmtss)

RSS = np.sum(RS)
TSS = np.sum(TS)
g.setStatistic('Method6-2DenseLayers', runName, 'ReE', 'R^2', (1 - RSS/TSS))

"""# ReHtilde"""

y_yhat, err = uts.y_yhat_errCFFs(data, results, 2)

pct_errs = ((y_yhat[:, 0] - y_yhat[:, 1])/y_yhat[:, 1])*100
g.setStatistic('Method6-2DenseLayers', runName, 'ReHtilde', 'MPE', np.mean(np.abs(pct_errs)))

RS = np.square((y_yhat[:, 0] - y_yhat[:, 1]))
TS = np.square((y_yhat[:, 0] - y_yhat[:, 0].mean()))
rmse = np.sqrt(np.mean(RS))
rmtss = np.sqrt(np.mean(TS))
g.setStatistic('Method6-2DenseLayers', runName, 'ReHtilde', 'RMSE', rmse)
g.setStatistic('Method6-2DenseLayers', runName, 'ReHtilde', 'RMTSS', rmtss)

RSS = np.sum(RS)
TSS = np.sum(TS)
g.setStatistic('Method6-2DenseLayers', runName, 'ReHtilde', 'R^2', (1 - RSS/TSS))


g.writeToFile(runName + '.json')

