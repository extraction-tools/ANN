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
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
import matplotlib.mlab as mlab

testmodel = tf.keras.models.load_model('/project/ptgroup/Devin/Neural_Network/Trained_Models_v12/trained_model_500K_v10.h5')

df = pd.read_csv('Sample_Data_v2/Sample_Data_Prediction_TE/Sample_Data_10k.csv')

y = df['P']
x = df.drop(['P'],axis=1)
# train_X, test_X, train_y, test_y = split_data(x,y)

X = df.drop(['P','SNR'],axis=1)
Y = testmodel.predict(X)
P = np.array(df['P'])
SNR = np.array(df['SNR'])
Y = Y.reshape((len(Y),))

err = ((P-Y)/(P))*100
# plt.scatter(len(err),err)
# plt.savefig('Accuracy.png')
result = pd.DataFrame(Y, columns ={'P'})
result = result.rename(columns={df.columns[0]:'P'},inplace=False)
result['P_True'] = P.tolist()
result['err'] = err.tolist()
# plt.plot(np.arange(len(result['P_True'])),result['P_True'],'.',label = 'True')
# result.to_csv('0p_v8.csv')

# ### Plotting ###

g = 0.05
s = 0.04
bigy=(3-s)**0.5
labelfontsize = 30

def cosal(x,eps):
    return (1-eps*x-s)/bigxsquare(x,eps)


def c(x):
    return ((g**2+(1-x-s)**2)**0.5)**0.5


def bigxsquare(x,eps):
    return (g**2+(1-eps*x-s)**2)**0.5


def mult_term(x,eps):
    return float(1)/(2*np.pi*np.sqrt(bigxsquare(x,eps)))


def cosaltwo(x,eps):
    return ((1+cosal(x,eps))/2)**0.5


def sinaltwo(x,eps):
    return ((1-cosal(x,eps))/2)**0.5


def termone(x,eps):
    return np.pi/2+np.arctan((bigy**2-bigxsquare(x,eps))/((2*bigy*(bigxsquare(x,eps))**0.5)*sinaltwo(x,eps)))


def termtwo(x,eps):
    return np.log((bigy**2+bigxsquare(x,eps)+2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps))/(bigy**2+bigxsquare(x,eps)-2*bigy*(bigxsquare(x,eps)**0.5)*cosaltwo(x,eps)))

def icurve(x,eps):
    return mult_term(x,eps)*(2*cosaltwo(x,eps)*termone(x,eps)+sinaltwo(x,eps)*termtwo(x,eps))

xvals = np.linspace(-6,6,500)
yvals = icurve(xvals,1)/10
yvals2 = icurve(-xvals,1)/10
x_arr = np.linspace(31.5,32.5,500)

result['P_diff'] = result['P'] - result['P_True']
x = np.array(result['P_diff'])*100

num_bins = 100
   
n, bins, patches = plt.hist(x, num_bins, 
                            density = True, 
                            color ='green',
                            alpha = 0.7)
   
(mu, sigma) = norm.fit(x)

# y = norm.pdf(bins, mu, sigma)
# l = plt.plot(bins, y, 'r--', linewidth=2)
  
# plt.plot(bins, y, '--', color ='black')
plt.hist(x,num_bins,density=True,color='green',alpha=0.7)
plt.xlabel('Percentage Difference (%)')
plt.ylabel('Count')
# plt.xlim([-5,5])
plt.title("Histogram of Percentage Difference: mu=%.3f, sigma=%.3f" %(mu, sigma))
# plt.title(r'$\mathrm{Histogram\ of\ I0%% Noise:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))

# plt.grid(True)
  
plt.savefig('Trained_Models_v12/Histogram_10K_v10.png')
acc = []
for i in range(0,200):
    p = P[i]
    p_pred = Y[i]
    accuracy = ((P[i] - Y[i])/P[i])*100
    acc.append(accuracy)
    snr = SNR[i]
    r = (np.sqrt(4-3*p**(2))+p)/(2-2*p)
    r_pred = (np.sqrt(4-3*p_pred**(2))+p_pred)/(2-2*p_pred)
    center = 250
    length = range(500)
    norm_array = []
    norm_array_pred = []
    for x in length:
        norm_array = np.append(norm_array,(x - center)*(12/500))  
        norm_array_pred = norm_array
    Iplus = icurve(norm_array,1)
    Iminus = icurve(norm_array,-1)
    ratio = Iminus/Iplus
    array = r*Iminus
    array_pred = r_pred*Iminus
    array_flipped = np.flip(array)
    array_pred_flipped = np.flip(array_pred)
    element_1 = array_flipped+Iminus
    sum_array = np.sum(array_flipped)*(12/500)
    element_2 = 1/sum_array
    element_3 = p
    element_1_pred = array_pred_flipped + Iminus
    sum_array_pred = np.sum(array_pred_flipped)*(12/500)
    element_2_pred = 1/sum_array_pred
    element_3_pred = p_pred
    result = element_1*element_2*element_3
    result_pred = element_1_pred*element_2_pred*element_3_pred
    result_new = result.reshape(500,)
    result_pred_new = result_pred.reshape(500,)
    arr = np.array(X.iloc[[i]]).reshape(500,)
    plt.figure()
    # plt.plot(norm_array,arr)
    plt.plot(norm_array,result_pred_new)
    plt.xlim(-3,3)
    plt.xlabel('R')
    plt.ylabel('Intensity')
    plt.legend(['True','Predicted'])
    plt.ylim(0,.015)
    plt.title("P_true:" + str(np.round(p*100,2)) + "\n P_pred:" + str(np.round(p_pred*100,2)) + "\n SNR:" + str(np.round(snr,3)))
    plt.savefig('Test_Plots_TE_v3/Model_Prediction__10K_V1_After_No'+ str(i) +'.png')
    plt.figure()
    # plt.plot(norm_array,result_new)
    arr = np.array(X.iloc[[i]]).reshape(500,)
    plt.plot(x_arr,arr)
    # plt.ylim(0,1)
    # plt.xlim(-3,3)
    plt.xlabel('Frequency')
    plt.ylabel('Intensity')
    # plt.legend(['True','Sample_Noise'])
    plt.title("P_pred:" + str(np.round(p_pred*100,2)) + "\n P_true:" + str(np.round(p*100,2)) + "\n SNR:" + str(np.round(snr,3)))
    plt.savefig('Test_Plots_TE_v3/_Model_Prediction_10K_V1_Before_No' + str(i) + '.png')
    
# plt.figure()
# plt.plot(acc)
# plt.savefig('Accuracy.png')
