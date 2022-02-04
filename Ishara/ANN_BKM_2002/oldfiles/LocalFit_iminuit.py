import pandas as pd
import numpy as np
import scipy.optimize as optimization
from tqdm.notebook import tqdm
import utilities as uts
#from BHDVCStf import BHDVCS
from TVA1_UU import TVA1_UU #modified bhdvcs file
import matplotlib.pyplot as plt
from iminuit import Minuit

filename="dvcs_xs_May-2021_342_sets.csv"


class DvcsData(object):
    def __init__(self, df):
        self.df=df.copy()
        self.X = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs']]
        self.XnoCFF = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs']]
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

    def sampleX(self):
        #return np.random.normal(self.y, self.erry)
        return self.X    
    
    def sampleY(self):
        #return np.random.normal(self.y, self.erry)
        return self.y

    def sampleErrY(self):
        #return np.random.normal(self.y, self.erry)
        return self.erry    
    
    def sampleWeights(self):
        return 1/self.erry

    
bhdvcs = TVA1_UU()
df = pd.read_csv(filename)
data = DvcsData(df)
XX = np.array(data.sampleX())
YY = np.array(data.sampleY())
eYY = np.array(data.sampleErrY())

num_data_sets=int(len(XX)/45)
fitting_sets=100


def Indv_Set(i):
    SetI = data.getSet(i)
    X  = np.array(SetI.XnoCFF)
    Y  = SetI.sampleY()
    eY = SetI.erry
    #pars = np.array([1, 1, 1])
    def TotalChi2XS(ReH, ReE, ReHtilde):
        tempTheory=bhdvcs.TotalUUXS(X, ReH, ReE, ReHtilde)
        return np.sum(((Y-tempTheory)/eY)**2)
    m = Minuit(TotalChi2XS,0,0,0)
    m.migrad()
    tempCFFs = m.values
    tempErrs = m.errors
    return (i,tempCFFs[0],tempErrs[0],tempCFFs[1],tempErrs[1],tempCFFs[2],tempErrs[2])



results_df = pd.DataFrame()

testresult=[]
setn=[]
tst_ReH=[]
tst_ReH_err=[]
tst_ReE=[]
tst_ReE_err=[]
tst_ReHtilde=[]
tst_ReHtilde_err=[]

for i in range(num_data_sets):
    testfit = Indv_Set(i)
    setn.append(int(np.array(testfit)[0]))
    tst_ReH.append(np.array(testfit)[1])
    tst_ReH_err.append(np.array(testfit)[2])
    tst_ReE.append(np.array(testfit)[3])
    tst_ReE_err.append(np.array(testfit)[4])
    tst_ReHtilde.append(np.array(testfit)[5])
    tst_ReHtilde_err.append(np.array(testfit)[6])
    print("Completed fitting to data set #",i)

    
results_df["#Set"]=setn
results_df["ReH"]=tst_ReH
results_df["sigmaReH"]=tst_ReH_err
results_df["ReE"]=tst_ReE
results_df["sigmaReE"]=tst_ReE_err
results_df["ReHtilde"]=tst_ReHtilde
results_df["sigmaReHtilde"]=tst_ReHtilde_err
testresult.append(results_df)



fitresults = pd.concat(testresult)
fitresults.to_csv('LocalFitResults_iminuit.csv')
    