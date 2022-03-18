import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from BHDVCStf import BHDVCS

bhdvcs = BHDVCS()


class DvcsData(object):
    def __init__(self, df):
        self.df = df
        self.X = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'ReH', 'ReE', 'ReHtilde', 'dvcs']]
        self.XnoCFF = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs', 'errF']] ## CHANGED
        self.CFFs = df.loc[:, ['ReH', 'ReE', 'ReHtilde']]
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
    
    def getAllKins(self, itemsInSets=36):
        return self.Kinematics.iloc[np.array(range(len(df)//itemsInSets))*itemsInSets, :]

    
def y_yhat_errCFFs(data, CFFdists, whichCFF):
    '''
    :param data: whole DvcsData
    :param CFFdists: (numSets, numReplicas, 3) shaped numpy array
    :param whichCFF: 0 for ReH, 1 for ReE, 2 for ReHtilde
    
    :returns: y_yhat as np array, err
    '''
    y_yhat = []
    err = []
    

    for whichSet in range(15):
        y_yhat.append([data.getSet(whichSet).X.iloc[0, whichCFF+7], CFFdists[whichSet, :, whichCFF].mean()])
        err.append(CFFdists[whichSet, :, whichCFF].std())
  
    return np.array(y_yhat), err


def y_yhat_errFs(CFFdists, data):
    '''
    :param CFFdists: (numSets, numReplicas, 3) shaped numpy array
    :param data: the whole DvcsData
    
    :returns: y_yhat as numpy array, errs
    '''
    y_yhat = []
    errs = []
    
    for i in range(CFFdists.shape[0]):
        sub = data.getSet(i)
        
        x = np.array(sub.XnoCFF.iloc[[18], :])
        reh = CFFdists[i, :, 0]
        ree = CFFdists[i, :, 1]
        rehtilde = CFFdists[i, :, 2]
        
        fs = bhdvcs.TotalUUXS(x, reh, ree, rehtilde).numpy()
        y_yhat.append([sub.y.iloc[18], fs.mean()])
        errs.append(fs.std())
    
    return np.array(y_yhat), errs

    
def evaluate(y_yhat):
    '''
    Provides a few evaluation statistics from an array of true values and predictions.
    
    :param y_yhat: numpy array with first column being true values and second being predicted values.
    '''
    pct_errs = ((y_yhat[:, 0] - y_yhat[:, 1])/y_yhat[:, 1])*100
    print('Mean percent error: ', np.mean(np.abs(pct_errs)))

    RS = np.square((y_yhat[:, 0] - y_yhat[:, 1]))
    TS = np.square((y_yhat[:, 0] - y_yhat[:, 0].mean()))
    rmse = np.sqrt(np.mean(RS))
    rmtss = np.sqrt(np.mean(TS))
    print('RMSE: ', rmse)
    print('RMSE w yhat=mean: ', rmtss)
    RSS = np.sum(RS)
    TSS = np.sum(TS)
    print('R-squared: ', (1 - RSS/TSS))
    plt.hist(np.array(pct_errs))
    plt.title('Histogram of Percent Errors')
    plt.show()
    
    
def plotError(y_yhat, errs, var_string, title=None):
    '''
    :param y_yhat: numpy array of what it sounds like
    :param errs: list or array of stds of variable
    :param var_string: string of which variable is being plotted
    '''
    assert len(y_yhat) == len(errs)
    
    fig, ax = plt.subplots()
    ax.errorbar(x=list(range(len(errs))),
                 y=list(map(lambda x: x[1], y_yhat)),
                yerr=errs,
                fmt='o',
                label="Estimated " + var_string)
    ax.plot(list(range(len(errs))),
             list(map(lambda x: x[0], y_yhat)),
             'ro', label="Actual " + var_string)
    
    ax.set_xlabel("Set#")
    ax.set_ylabel(var_string)
    if title != None:
        ax.set_title(title)
        
    ax.legend()
    
    plt.show()
    
    
class TotalUUXSlayer(tf.keras.layers.Layer):
    '''
    class for incorporating TotalUUXS function into tensorflow layer
    '''
    def __init__(self):
        super(TotalUUXSlayer, self).__init__(dtype='float64')
        self.F = BHDVCS()
    def call(self, inputs):
        return self.F.TotalUUXS(inputs[:, :8], inputs[:, 9], inputs[:, 10], inputs[:, 11]) + [np.random.normal(0, inputs[i, 8]) for i in range(len(inputs))] ## NEW LINE
    
    
def cffs_from_globalModel(model, kinematics, numHL=1):
    '''
    :param model: the model from which the cffs should be predicted
    :param kinematics: the kinematics that should be used to predict
    :param numHL: the number of hidden layers:
    '''
    subModel = tf.keras.backend.function(model.layers[0].input, model.layers[numHL+2].output)
    return subModel(np.asarray(kinematics)[None, 0])[0]


class f1_f2(object):
    '''
    Calculate f1 and f2
    '''
    
    def __init__(self):
        # this could be refactored to include the constants that are written explicitly below
        pass
    
    @staticmethod
    def ffGE(t):
        GE = 1.0 / ( 1.0 + ( -t / 0.710649 ) ) / ( 1.0 + ( -t / 0.710649 ) )
        return GE

    @staticmethod
    def ffGM(t):
        shape = f1_f2.ffGE(t)
        GM0 = 2.792847337
        return GM0*shape
    
    @staticmethod
    def ffF2(t):
        f2 = (f1_f2.ffGM(t) - f1_f2.ffGE(t)) / (1. - t / (4.*.938272*.938272));
        return f2
    
    @staticmethod
    def ffF1(t):
        f1 = f1_f2.ffGM(t) - f1_f2.ffF2(t)
        return f1
    
    @staticmethod
    def ffGA(t):
        ga = 1.2695
        ma = 1.026
        part = t/(ma*ma);
        dif = (1-part)*(1-part);
        GA = ga/dif
        return GA
    
        
