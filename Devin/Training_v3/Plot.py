import pandas as pd
import numpy as np
import h5py as h
from matplotlib import pyplot as plt
import random
from matplotlib.offsetbox import AnchoredText
from scipy.stats import norm
import matplotlib.mlab as mlab

df = pd.read_csv('Test.csv')

y = df['P']
x = df.drop(['P'],axis=1)

P = np.array(df['P'])
SNR = np.array(df['SNR'])

X = df.drop(['P','SNR'],axis=1) 


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

for i in range(0,len(y)):
    p = np.round(P[i],6)
    snr = np.round(SNR[i],6)
    r = (np.sqrt(4-3*p**(2))+p)/(2-2*p)
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
    array_flipped = np.flip(array)
    element_1 = array_flipped+Iminus
    sum_array = np.sum(array_flipped)*(12/500)
    element_2 = 1/sum_array
    element_3 = p
    result = element_1*element_2*element_3
    result_new = result.reshape(500,)
    # plt.plot(norm_array,result_new)
    arr = np.array(X.iloc[[i]]).reshape(500,)
    plt.figure()
    plt.plot(x_arr,arr)
    # plt.ylim(0,1)
    # plt.xlim(-3,3)
    plt.xlabel('Frequency')
    plt.ylabel('Intensity')
    # plt.legend(['True','Sample_Noise'])
    plt.title("P_true:" + str(np.round(p*100,6)) + "\n SNR:" + str(snr))
    plt.savefig('Test_Plot/Test_No' + str(i) + '.png')