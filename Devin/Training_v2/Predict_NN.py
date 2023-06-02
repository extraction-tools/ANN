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

testmodel = tf.keras.models.load_model('/project/ptgroup/Devin/Neural_Network/Trained_Models_v2/trained_model_1M_redux_v12.h5')

df = pd.read_csv('Sample_Data/Sample_Data_100k/Sample_Data_100k.csv')

y = df['P']
x = df.drop(['P'],axis=1)
# train_X, test_X, train_y, test_y = split_data(x,y)

X = df.drop(['P'],axis=1)
Y = np.around(testmodel.predict(X),decimals=2)
P = np.array(df['P'])
Y = Y.reshape((len(Y),))
err = (abs(P-Y)/(P))*100
result = pd.DataFrame(Y, columns ={'P'})
result = result.rename(columns={df.columns[0]:'P'},inplace=False)
result['P_True'] = P.tolist()
result['err'] = err.tolist()
# plt.plot(np.arange(len(result['P_True'])),result['P_True'],'.',label = 'True')
# result.to_csv('0p_v8.csv')

### Plotting ###

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


result['P_diff'] = result['P'] - result['P_True']
x = np.array(result['P_diff'])*100

num_bins = 100
   
n, bins, patches = plt.hist(x, num_bins, 
                            density = True, 
                            color ='green',
                            alpha = 0.7)
   
(mu, sigma) = norm.fit(x)

y = norm.pdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
  
# plt.plot(bins, y, '--', color ='black')
  
plt.xlabel('Percentage Difference')
plt.ylabel('Count')
plt.title("Histogram of 0-25%% Noise: mu=%.3f, sigma=%.3f" %(mu, sigma))
# plt.title(r'$\mathrm{Histogram\ of\ I0%% Noise:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))

plt.grid(True)
  
plt.savefig('Histogram_v12.png')

# for i in range(0,len(Y)):
#     p = P[i]
#     p_pred = Y[i]
#     r = (np.sqrt(4-3*p**(2))+p)/(2-2*p)
#     r_pred = (np.sqrt(4-3*p_pred**(2))+p_pred)/(2-2*p_pred)
#     center = 250
#     length = range(501)
#     norm_array = []
#     norm_array_pred = []
#     for x in length:
#         norm_array = np.append(norm_array,(x - center)*(12/500))  
#         norm_array_pred = norm_array
#     Iplus = icurve(norm_array,1)
#     Iminus = icurve(norm_array,-1)
#     ratio = Iminus/Iplus
#     array = r*Iminus
#     array_pred = r_pred*Iminus
#     array_flipped = np.flip(array)
#     array_pred_flipped = np.flip(array_pred)
#     element_1 = array_flipped+Iminus
#     sum_array = np.sum(array_flipped)*(12/500)
#     element_2 = 1/sum_array
#     element_3 = p
#     element_1_pred = array_pred_flipped + Iminus
#     sum_array_pred = np.sum(array_pred_flipped)*(12/500)
#     element_2_pred = 1/sum_array_pred
#     element_3_pred = p_pred
#     result = element_1*element_2*element_3
#     result_pred = element_1_pred*element_2_pred*element_3_pred
#     plt.figure()
#     plt.plot(norm_array,result)
#     plt.plot(norm_array,result_pred)
#     plt.ylim(0,1)
#     plt.xlim(-3,3)
#     plt.xlabel('R')
#     plt.ylabel('Intensity')
#     plt.legend(['True','Predicted'])
#     at = AnchoredText("Err:" + str(err[i]) + "\n P_true:" + str(p) + "\n P_pred:" + str(p_pred),loc='upper left', prop=dict(size=8), frameon=True,)
#     at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#     plt.text(0.001, 0.9, "Err:" + str(err[i]) + "\n P_true:" + str(p) + "\n P_pred:" + str(p_pred), bbox=dict(facecolor='white', alpha=0.5))
#     plt.savefig('Plots/Prediction_Plots/Model_Prediction_No'+ str(i) +'_after_5p.png')
#     plt.figure()
#     plt.plot(norm_array,result)
#     plt.plot(norm_array,np.array(X.iloc[[i]]))
#     plt.ylim(0,1)
#     plt.xlim(-3,3)
#     plt.xlabel('R')
#     plt.ylabel('Intensity')
#     plt.legend(['True','Sample_Noise'])
#     plt.text(0.001, 0.9, "Err:" + str(err[i]) + "\n P_true:" + str(p) + "\n P_pred:" + str(p_pred), bbox=dict(facecolor='white', alpha=0.5))
#     plt.savefig('Plots/Prediction_Plots_Model_Before_No' + str(i) + '.png')
