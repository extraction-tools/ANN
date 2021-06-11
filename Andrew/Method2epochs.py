import pandas as pd
import numpy as np
#from tqdm.notebook import tqdm
import tensorflow as tf

from TVA1_UU import TVA1_UU as BHDVCS #modified bhdvcs file
import utilities as uts #general utilities that are useful for all methods
import sys


# General global variable definitions

bhdvcs = BHDVCS()
df = pd.read_csv("dvcs_xs_May-2021_342_sets.csv")
data = uts.DvcsData(df)
#numSets = 15
numReplicas = 1000


def cffs_from_globalModel(model, kinematics):
    subModel = tf.keras.backend.function(model.layers[0].input, model.layers[3].output)
    return subModel(np.asarray(kinematics)[None, 0])[0]


class TotalUUXS(tf.keras.layers.Layer):
    def __init__(self):
        super(TotalUUXS, self).__init__(dtype='float64')
        self.F = BHDVCS()
    def call(self, inputs):
        return self.F.TotalUUXS(inputs[:, :8], inputs[:, 8], inputs[:, 9], inputs[:, 10])

kinematics = tf.keras.Input(shape=(4))
x = tf.keras.layers.Dense(20, activation="tanh")(kinematics)
outputs = tf.keras.layers.Dense(3)(x) #three output nodes for ReH, ReE, ReHtilde
noncffInputs = tf.keras.Input(shape=(8))
totalUUXSInputs = tf.keras.layers.concatenate([noncffInputs, outputs])
F = uts.TotalUUXSlayer()(totalUUXSInputs) # incorporate cross-sectional function

globalModel = tf.keras.Model(inputs=[kinematics, noncffInputs], outputs=F, name="GlobalModel")

globalModel.compile(
    optimizer = tf.keras.optimizers.Adam(.02),
    loss = tf.keras.losses.MeanSquaredError(),
)

Wsave2 = globalModel.get_weights()


# Training

numSamples = 1000 # Number of Replicas

i = int(sys.argv[1])
print("HELLO"+)
epoch = int(sys.argv[2])
print("THERE"+str(epoch))

setI = data.getSet(i)

results = pd.DataFrame({
    "ReH": np.zeros(numSamples),
    "ReE": np.zeros(numSamples),
    "ReHtilde": np.zeros(numSamples),
    "SetNum": np.repeat(i, numSamples)
    })


for sample in range(numSamples):

    chkpt_path = 'best-network' + str(i) + '_' + str(epoch) + '.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=chkpt_path,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    globalModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(),
                epochs=epoch, verbose=0,
                callbacks=[model_checkpoint_callback])

    globalModel.load_weights(chkpt_path)

    cffs = cffs_from_globalModel(globalModel, setI.Kinematics)

    for num, cff in enumerate(['ReH', 'ReE', 'ReHtilde']):
        results.loc[sample, cff] = cffs[num]

results.to_csv("/home/atz6cq/method2/Results"+ str(i) + "_" + str(epoch) + ".csv")