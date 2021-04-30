

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler

from BHDVCStf import BHDVCS #modified bhdvcs file
import utilities as uts


numSets = 15
numReplicas = 500

df = pd.read_csv("dvcs_xs_newsets_genCFFs.csv")
data = uts.DvcsData(df)

localFits = np.fromfile('replicas500.txt')  # can either be produced in situ or brought with
localFits = replicas.reshape(numSets, numReplicas, 3)

i = int(sys.argv[1])

class step3sim(object):
    def __init__(self, X, replicas, whichSet, numSamples=10000):
        # sample range of 10,000, with replicas repeated and appended to fill
        
        self.X = X
        self.cffs = replicas[whichSet, :, :] # 2d numpy array of shape (replicas, cffs)
        self.whichSet = whichSet        
        self.numSamples = numSamples
        
        self.kins = X.loc[whichSet*36, ['k', 'QQ', 'x_b', 't']] #pandas array of kins for set
        self.dvcs = X.loc[whichSet*36, 'dvcs']
            
    
    def _getKinRange(self, whichKin):
        '''
        :param whichKin: which kinematic to compute range for
        
        :returns: [low end of range, high end of range]
        '''
        krange = np.unique(self.X[whichKin].sort_values())
        idx = np.where(krange == self.kins[whichKin])[0][0]
        
        if idx == 0:  # if kin is lowest, then use for low end (kin - (high - kin) = 2*kin - high)
            return [self.kins[whichKin]*2 - krange[idx + 1], krange[idx + 1]]
        elif idx == (len(krange) - 1):   # if kin is highest, then use for high end (kin + (kin - low) = 2*kin - low)
            return [krange[idx - 1], 2*self.kins[whichKin] - krange[idx - 1]]
        else:
            return [krange[idx - 1], krange[idx + 1]]


    
    def addFs(self, sd):
        '''
        :param sd: dataframe with all necessary variables from which F can be computed
        
        :returns: same dataframe but with F column added
        '''
        bhdvcs = BHDVCS()
        sd['F'] = bhdvcs.TotalUUXS(np.array(sd[['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs']]),
                                  np.array(sd['ReH']), np.array(sd['ReE']), np.array(sd['ReHtilde']))
        return sd    

    
    def getSimData(self):
        to_ret = {key: [] for key in self.X.columns} # compose dataframe with each necessary variable
        
        for kin in ['k', 'QQ', 'x_b', 't']: # uniformly sample from kin range for each kinematic
            lohi = self._getKinRange(kin)
            to_ret[kin] = np.random.uniform(lohi[0], lohi[1], self.numSamples)
        
        to_ret['F1'] = uts.f1_f2.ffF1(to_ret['t'])
        to_ret['F2'] = uts.f1_f2.ffF1(to_ret['t'])
                                

        for i, cff in enumerate(['ReH', 'ReE', 'ReHtilde']): # normally sample cffs using local sets to parameterize
            #can also be uniform 
            to_ret[cff] = np.random.normal(self.cffs[:, i].mean(), self.cffs[:, i].std(), size=self.numSamples)

        
        to_ret['dvcs'] = np.repeat(self.dvcs, self.numSamples)
        
        phis = np.repeat(np.linspace(0, 350, 36), self.numSamples//36 + 1)
        to_ret['phi_x'] = phis[:self.numSamples]
        
        sd = pd.DataFrame(to_ret)
        sd = self.addFs(sd)
        return sd
    

t = step3sim(data.X, localFits, 0, 537)
test = t.getSimData()

(test['F'] < 0).mean()

np.unique(data.X['k'])

test

def fit_pred_kinSet(whichSet, data, localFits, numReplicas):
    
    s3 = step3sim(data.X, localFits, whichSet)
    locData = s3.getSimData()
    
    locData = locData.dropna() # some F values are NA
    
    rescaler = MinMaxScaler()
    rescaler = rescaler.fit(locData[['k', 'QQ', 'x_b', 't', 'F']])
    X = rescaler.transform(locData[['k', 'QQ', 'x_b', 't', 'F']])
    y = locData[['ReH', 'ReE', 'ReHtilde']]
    
    
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, input_shape=(5,), activation="elu", kernel_initializer="he_normal"),
    tf.keras.layers.Dense(20, activation="elu", kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(3)
    ])
    
    model.compile(
    optimizer = tf.keras.optimizers.Nadam(.01),
    loss = tf.keras.losses.MeanSquaredError()
    )
    
    model.fit(X, y, epochs=100, verbose=0)
        

    #to_pred = np.tile(np.array(data.Kinematics.loc[whichSet*36 + 18, :]), (100, 1)) 

    to_pred = pd.concat([data.Kinematics.loc[[whichSet*36 + 18], :] for _ in range(numReplicas)]) #phi=180
    to_pred['F'] = [data.getSet(whichSet).sampleY()[18] for _ in range(numReplicas)] #phi=180
    
    to_pred = rescaler.transform(to_pred)
    
    #yhat = np.stack([model(to_pred, training=True) for sample in range(100)]).reshape()
    
    return model.predict(to_pred)

def produceResults(data, localFits, numSets, numReplicas):
    '''
    data: a tensorflow neural network model
    X: [standardized kins, xnocff]
    orig_weights: the original weights from when the model was created (used to reset model after it has been trained)
    numSets: the number of kinematic sets
    numReplicas: the number of replicas
    
    returns: np array of cff predictions of shape (numSets, numReplicas, numCFFs)
    '''
    by_set = []
    for i in tqdm(range(numSets)):
        by_set.append(fit_pred_kinSet(i, data, localFits, numReplicas).tolist())

    return np.array(by_set)

results = produceResults(data, localFits, numSets, numReplicas)

results.to_csv("/home/ncn6mq/Results"+ str(i) + ".csv")
