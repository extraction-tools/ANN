import numpy as np
import utilities as uts
from TVA1_UU import TVA1_UU as BHDVCS
import pandas as pd
import sys
data = pd.read_csv('dvcs_xs_May-2021_342_sets_with_trueCFFs.csv')
bhdvcs = BHDVCS()

class DvcsData(object):
    def __init__(self, df):
        self.df = df
        self.X = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'ReH', 'ReE', 'ReHTilde', 'dvcs']]
        self.XnoCFF = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs']]
        #self.X = self.XnoCFF ReH,ReE and ReHtilde no longer in new data
        self.CFFs = df.loc[:, ['ReH', 'ReE', 'ReHTilde']] # ReH,ReE and ReHtilde no longer in new data
        self.y = df.loc[:, 'F']
        self.Kinematics = df.loc[:, ['k', 'QQ', 'x_b', 't']]
        self.erry = df.loc[:, 'sigmaF']
        
    def __len__(self):
        return len(self.X)
    
    def getSet(self, setNum, itemsInSet=45):
        pd.options.mode.chained_assignment = None
        subX = self.X.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1, :]
        subX['F'] = self.y.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1]
        subX['sigmaF'] = self.erry.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1]
        pd.options.mode.chained_assignment = 'warn'
        return DvcsData(subX)
    
    def sampleY(self):
        return np.random.normal(self.y, self.erry)
    
    def sampleWeights(self):
        return 1/self.erry
    
    def getAllKins(self, itemsInSets=45):
        return self.Kinematics.iloc[np.array(range(len(df)//itemsInSets))*itemsInSets, :]

dvcsdata = DvcsData(data)

def errFunc(data, cff):
    cff = np.reshape(cff, (-1,3))
    ReH = cff[:,0]
    ReE = cff[:,1]
    ReHT= cff[:,2]
    
    #print(np.shape(data.XnoCFF))
    dats = data.X
    k = np.array(dats['k'])
    qq = np.array(dats['QQ'])
    xb = np.array(dats['x_b'])
    t = np.array(dats['t'])
    phi = np.array(dats['phi_x'])
    F1 = np.array(dats['F1'])
    F2 = np.array(dats['F2'])
    const = np.array(dats['dvcs'])
    xdat = np.transpose(np.array([phi, k, qq, xb, t, F1, F2, const]))
    #print(np.shape(xdat))
    # idk why i need to use xdat instead of XnoCFF
    err = np.array([])
    for i in range(len(ReH)):
        calcF = bhdvcs.TotalUUXS(xdat,ReH[i],ReE[i],ReHT[i])
        err = np.append(err,np.mean(np.power(np.subtract(data.y,calcF),2)))
    return err
    

def nm(sets, epochs):
    alpha = 1
    gamma = 2
    p = .5
    sigma = .5
    #startCFFs = np.array([[1.,1,1],[1,1,2],[1,2,1],[2,2,2]])
    startCFFs = np.random.random((4,3))*20-10
    for epoch in range(epochs):
        mse = errFunc(dvcsdata.getSet(sets),startCFFs)
        sort = np.argsort(mse)
        centroid = np.mean([startCFFs[i] for i in sort[0:-1]],axis = 0)
        centroidmse = errFunc(dvcsdata.getSet(sets),centroid)
        reflect = centroid - alpha * (centroid - startCFFs[sort[-1]])
        reflectmse = errFunc(dvcsdata.getSet(sets),reflect)
        if (mse[sort[0]] <= reflectmse) and (reflectmse < mse[sort[-2]]):
            startCFFs[sort[-1]] = reflect
            continue
        if (reflectmse < mse[sort[0]]):
            expand = centroid + gamma * (reflect - centroid)
            expandmse = errFunc(dvcsdata.getSet(sets),expand)
            if expandmse < reflectmse:
                startCFFs[sort[-1]] = expand
                continue
            else:
                startCFFs[sort[-1]] = reflect
                continue
        #if (reflectmse >= startCFFs[sort[-1]]):
        contract = np.array(centroid + p * (startCFFs[sort[-1]] - centroid))
        contractmse = errFunc(dvcsdata.getSet(sets),contract)
        if contractmse < mse[sort[-1]]:
            startCFFs[sort[-1]] = contract
            continue
        for i in sort[1:]:
            startCFFs[i] = startCFFs[sort[0]] + sigma * (startCFFs[i] - startCFFs[sort[0]])
    
    return startCFFs


epoch = int(sys.argv[1])
setno = int(sys.argv[2])
replica = int(sys.argv[3])

points = nm(setno,epoch)

results = pd.DataFrame(points, columns=['ReH', 'ReE', 'ReHT'])
results['index'] = range(0, len(results))
results['replica'] = replica

results.to_csv("/home/atz6cq/nm/" + "Results"+ str(setno) + "_" + str(epoch) +  "/Results"+ str(setno) + "_" + str(epoch) + "_" + str(replica) + ".csv")