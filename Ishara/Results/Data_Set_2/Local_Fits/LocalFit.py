import pandas as pd
import numpy as np
import scipy.optimize as optimization
from tqdm.notebook import tqdm
import utilities as uts
#from BHDVCStf import BHDVCS
from TVA1_UU import TVA1_UU #modified bhdvcs file
import matplotlib.pyplot as plt

#### Provid the data set .csv file name, number of data sets (# of different kinematics) and number of replicas here ####
filename="dvcs_xs_May-2021_342_sets.csv"
numSets = 342
numReplicas = 100


#####
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

    def sampleY(self):
        return np.random.normal(self.y, self.erry)

    def sampleWeights(self):
        return 1/self.erry


#bhdvcs = BHDVCS()
bhdvcs = TVA1_UU()
df = pd.read_csv(filename)
data = DvcsData(df)


def SelectSet(setnum):
    tempdf=pd.read_csv(filename)
    full_data_size=len(tempdf["#Set"])
    temp_k=[]
    temp_QQ=[]
    temp_xb=[]
    temp_t=[]
    temp_phi=[]
    temp_F=[]
    temp_dF=[]
    print(full_data_size)
    for i in range(0,full_data_size):
        if(tempdf["#Set"][i]==setnum):
            temp_k.append(tempdf["k"][i])
            temp_phi.append(tempdf["phi_x"][i])
    return temp_phi

   
def produceResults(data, numSets, numReplicas):
    {"#Set":[],"k":[],"QQ":[],"x_b":[],"t":[],"phi_x":[],"F_data":[],"sigmaF_data":[],"F_cal":[],"sigmaF_cal":[],"ReH":[],"sigmaReH":[],"ReE":[],"sigmaReE":[],"ReHtilde":[],"sigmaReHtilde":[]}
    results = []
    for i in tqdm(range(numSets)):
        CFFreplicas = []
        Temp_Fcals=[]
        Temp_H=[]
        Temp_E=[]
        Temp_Htilde=[]
        Temp_phi=[]
        seti = data.getSet(i) # DvcsData object of kinematic set i
        X = np.array(seti.XnoCFF) # the kinematics and all variables necessary to compute 
        sigma = seti.erry # error in F
        pars = np.array([1, 1, 1])
        #setdf= []
        Tempdf=seti.df.copy()
        for j in range(numReplicas):
            #Tempdf=seti.df.copy()
            #Tempdf["Rep_num"]=j
            #setdf.append(Tempdf)
            y = seti.sampleY()
            cff, cffcov = optimization.curve_fit(bhdvcs.TotalUUXS, X, y, pars, sigma,method='lm')
            Temp_Fcals.append(bhdvcs.TotalUUXS(np.array(data.getSet(i).X),cff[0],cff[1],cff[2]))
            Temp_H.append(cff[0])
            Temp_E.append(cff[1])
            Temp_Htilde.append(cff[2])
        #setdf=pd.concat(setdf)
        Tempdf["F_cal"]=np.mean(np.array(Temp_Fcals),axis=0)
        Tempdf["sigmaF_cal"]=np.std(np.array(Temp_Fcals),axis=0)
        Tempdf["ReH"]=np.mean(Temp_H)
        Tempdf["sigmaReH"]=np.std(Temp_H)
        Tempdf["ReE"]=np.mean(Temp_E)
        Tempdf["sigmaReE"]=np.std(Temp_E)
        Tempdf["ReHtilde"]=np.mean(Temp_Htilde)
        Tempdf["sigmaReHtilde"]=np.std(Temp_Htilde)
        Tempdf["#Set"]=i
        results.append(Tempdf)
    return pd.concat(results)
        


def Histogram_Data(fitname,setnum,cffnum):
    Cffdat=[]
    LL=len(fitname[setnum])
    for i in range(LL):
        Cffdat.append(fitname[setnum][i][cffnum])
    return np.array(Cffdat)


test_fit=produceResults(data, numSets, numReplicas)
test_fit.to_csv('LocalFitResults_curvefit.csv')


