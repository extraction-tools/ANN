import numpy as np
import utilities as uts
from TVA1_UU import TVA1_UU as BHDVCS
import pandas as pd
import sys
data = pd.read_csv('dvcs_xs_May-2021_342_sets_with_trueCFFs.csv')
bhdvcs = BHDVCS()

epochno = 0
setno = 0
replicas = 0
alpha = 1
gamma = 2
rho = .5
sigma = .5
STDEVS = np.array([.1,.1.,1])

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
    

#version for configs that returns a df with all the cffs for each iteration
def nm(sets, epochs, startCFF):
    totCFF = pd.DataFrame()
    CFFs = startCFF.copy()
    lastmove = ""
    
    STDEVS = np.array([.05,.15,.05])
    for epoch in range(epochs):
        mse = errFunc(dvcsdata.getSet(sets),CFFs)
        sort = np.argsort(mse)
        
        ranks = np.empty_like(sort)
        ranks[sort] = np.arange(len(mse))
        df = pd.DataFrame(CFFs, columns = ['ReH', 'ReE', 'ReHT'])
        df['lastMove'] = lastmove
        df['epoch'] = epoch
        df['error'] = mse
        df['rank'] = ranks
        totCFF = totCFF.append(df)
        
        if np.max(np.std(CFFs)>STDEVS):
            CFFs = np.random.random((4,3))*2-1+CFFs
            lastmove="reshuffle"
            continue
        
        centroid = np.mean([CFFs[i] for i in sort[0:-1]],axis = 0)
        centroidmse = errFunc(dvcsdata.getSet(sets),centroid)
        reflect = centroid - alpha * (centroid - CFFs[sort[-1]])
        reflectmse = errFunc(dvcsdata.getSet(sets),reflect)
        if (mse[sort[0]] <= reflectmse) and (reflectmse < mse[sort[-2]]):
            CFFs[sort[-1]] = reflect
            lastmove = "reflection"
            continue
        if (reflectmse < mse[sort[0]]):
            expand = centroid + gamma * (reflect - centroid)
            expandmse = errFunc(dvcsdata.getSet(sets),expand)
            if expandmse < reflectmse:
                CFFs[sort[-1]] = expand
                lastmove = "expansion"
                continue
            else:
                CFFs[sort[-1]] = reflect
                lastmove = "reflected-exp"
                continue
        #if (reflectmse >= CFFs[sort[-1]]):
        contract = np.array(centroid + rho * (CFFs[sort[-1]] - centroid))
        contractmse = errFunc(dvcsdata.getSet(sets),contract)
        if contractmse < mse[sort[-1]]:
            CFFs[sort[-1]] = contract
            lastmove = "contraction"
            continue
        for i in sort[1:]:
            CFFs[i] = CFFs[sort[0]] + sigma * (CFFs[i] - CFFs[sort[0]])
        lastmove = "shrink"
    
    return totCFF

def readConfig(filename, lineno):
    global epochno, setno, replicas, alpha, gamma, rho, sigma, startCFFs
    configs = pd.read_csv(filename)
    config = configs.iloc[[lineno]]
    del configs
    if 'alpha' in config.columns:
        alpha = config['alpha'][lineno]
    if 'gamma' in config.columns:
        gamma = config['gamma'][lineno]
    if 'rho' in config.columns:
        rho = config['rho'][lineno]
    if 'sigma' in config.columns:
        sigma = config['sigma'][lineno]
    if 'ReH' in config.columns:
        startCFFs[0] = config['ReH'][lineno] 
    if 'ReE' in config.columns:
        startCFFs[1] = config['ReE'][lineno]
    if 'ReHT' in config.columns:
        startCFFs[2] = config['ReHT'][lineno]
    if 'epochs' in config.columns:
        epochno = config['epochs'][lineno]
    if 'replicas' in config.columns:
        replicas = config['replicas'][lineno]
    if 'set' in config.columns:
        setno = config['set'][lineno]

startCFFs = np.random.random((4,3))*20-10
filename = str(sys.argv[1])
linenum = int(sys.argv[2])
readConfig(filename, linenum)


results = pd.DataFrame()
for replica in range(replicas):
    startCFFs = np.random.random((4,3))*20-10
    result = nm(setno,epochno,startCFFs)
    #result = pd.DataFrame(points, columns=['ReH', 'ReE', 'ReHT'])
    #result['index'] = range(0, len(result))
    result['replica'] = replica
    result['set'] = setno
    results = results.append(result)

results.to_csv("/home/atz6cq/nm/" + "ResultsConfig" +  "/Results"+ str(linenum) + ".csv")