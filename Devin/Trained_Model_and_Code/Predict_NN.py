import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import h5py as h
from matplotlib import pyplot as plt
from tensorflow.keras import regularizers
from sklearn.utils import class_weight
import random
from keras.layers import GaussianNoise

testmodel = tf.keras.models.load_model('/project/ptgroup/Devin/Neural_Network/Trained_Models/trained_model_5M.h5')

df = pd.read_csv('Sample_Data/Sample_Data_5M_Noisy/Sample_Data_5M_noisy.csv')

def split_data(X, y, split=0):
    temp = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
    trn_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})
    
    tst_y = y[temp]
    trn_y = y.drop(temp)
    
#     tst_err = err[tstidxs]
#     trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y

y = df['P']
x = df.drop(['P'],axis=1)
# train_X, test_X, train_y, test_y = split_data(x,y)

X = df.drop(['P'],axis=1)
Y = np.around(testmodel.predict(X),decimals=2)
P = np.array(df['P'])
Y = Y.reshape((len(Y),))
result = pd.DataFrame(Y, columns ={'P'})
result = result.rename(columns={df.columns[0]:'P'},inplace=False)
result['P_True'] = P.tolist()
plt.plot(np.arange(len(result['P_True'])),result['P_True'],'.',label = 'True')
result.to_csv('Test.csv')
plt.plot(np.arange(len(result['P'])),result['P'],'.')
plt.xlabel('Data Point ID')
plt.ylabel('Predicted Polarization')
plt.ylim([0,1])
plt.legend(['Pre','True'])
plt.savefig("Plots/NN_Example_5M.jpeg")
upper_lim = 50
lower_lim = 0
# result['P'].plot.bar(figsize=(100,20),alpha=1,color='red',label='Predicted')
# result['P_True'].plot.bar(figsize=(100,20),alpha=1,color='black',label='True')
# plt.xlabel('Data Point',fontsize=64)
# plt.ylabel('Polarization',fontsize=64)
# plt.ylim([0,1])
# plt.savefig('Plots/NN_Histogram_5M.jpeg')

upper_lim = 50
plt.figure()
plt.plot(np.arange(len(result['P'][lower_lim:upper_lim])),result['P'][lower_lim:upper_lim],'.',label="Pre")
plt.xlabel('Data Point ID')
plt.ylabel('Predicted Polarization')
plt.plot(np.arange(len(result['P_True'][lower_lim:upper_lim])),result['P_True'][lower_lim:upper_lim],'.',label = 'True')
plt.legend(['True','Pre'])
plt.title('Data Point ID: 0 - 50')
plt.ylim([0,1])
plt.savefig('Plots/NN_Example_5k_5M.jpeg')
