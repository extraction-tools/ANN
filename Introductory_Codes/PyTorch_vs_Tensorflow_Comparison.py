import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#### Please define the path to the data files ####
BKM_Pytorch_results = 'BKM_PyTorch/test_v1'
BKM_tf_results = 'BKM_tf/test_v1'

BKM_pt_folders=os.listdir(BKM_Pytorch_results)
BKM_tf_folders=os.listdir(BKM_tf_results)
##################################################

Kinematic_Sets=5
Replicas=100


def CFF_vals(CFF_ID,Kin_Set,Reps):
    CFF_PT=[]
    CFF_TF=[]
    for i in range(1,Reps+1):
        tempdf_BKM_pt=pd.read_csv(BKM_Pytorch_results+'/'+str(i)+'/bySetCFFs.csv',header=None, prefix="var")
        tempdf_BKM_tf=pd.read_csv(BKM_tf_results+'/'+str(i)+'/bySetCFFs.csv',header=None, prefix="var")
        CFF_PT.append(tempdf_BKM_pt['var'+str(CFF_ID)][Kin_Set])
        CFF_TF.append(tempdf_BKM_tf['var'+str(CFF_ID)][Kin_Set])
    return CFF_PT,CFF_TF


def CFF_Hist(CFF_ID,CFF_Name,Kin_Set,Reps):
    h1=CFF_vals(CFF_ID,Kin_Set,Reps)[0]
    h2=CFF_vals(CFF_ID,Kin_Set,Reps)[1]
    plt.hist(h1, label=str(CFF_Name)+' Pt', color = 'g',alpha = 0.5)
    plt.hist(h2, label=str(CFF_Name)+' Tf', color = 'r',alpha = 0.5)
    plt.legend()

    
f1=plt.figure(1,figsize=(15,10))
######### Kin 1 ###########
plt.subplot(5,4,1)
CFF_Hist(1,'ReH',1,Replicas)
plt.subplot(5,4,2)
CFF_Hist(2,'ReE',1,Replicas)
plt.subplot(5,4,3)
CFF_Hist(3,'ReHt',1,Replicas)
plt.subplot(5,4,4)
CFF_Hist(4,'c1',1,Replicas)
######### Kin 2 ###########
plt.subplot(5,4,5)
CFF_Hist(1,'ReH',2,Replicas)
plt.subplot(5,4,6)
CFF_Hist(2,'ReE',2,Replicas)
plt.subplot(5,4,7)
CFF_Hist(3,'ReHt',2,Replicas)
plt.subplot(5,4,8)
CFF_Hist(4,'c1',2,Replicas)
######### Kin 3 ###########
plt.subplot(5,4,9)
CFF_Hist(1,'ReH',3,Replicas)
plt.subplot(5,4,10)
CFF_Hist(2,'ReE',3,Replicas)
plt.subplot(5,4,11)
CFF_Hist(3,'ReHt',3,Replicas)
plt.subplot(5,4,12)
CFF_Hist(4,'c1',3,Replicas)
######### Kin 4 ###########
plt.subplot(5,4,13)
CFF_Hist(1,'ReH',4,Replicas)
plt.subplot(5,4,14)
CFF_Hist(2,'ReE',4,Replicas)
plt.subplot(5,4,15)
CFF_Hist(3,'ReHt',4,Replicas)
plt.subplot(5,4,16)
CFF_Hist(4,'c1',4,Replicas)
######### Kin 4 ###########
plt.subplot(5,4,17)
CFF_Hist(1,'ReH',5,Replicas)
plt.subplot(5,4,18)
CFF_Hist(2,'ReE',5,Replicas)
plt.subplot(5,4,19)
CFF_Hist(3,'ReHt',5,Replicas)
plt.subplot(5,4,20)
CFF_Hist(4,'c1',5,Replicas)
f1.tight_layout()
f1.savefig('Comparison_Plots_PyTorch_vs_Tensorflow.pdf')    