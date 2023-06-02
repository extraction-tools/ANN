import pandas as pd
import numpy as np
import random

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

center = 250
length = range(501)
norm_array = []
for x in length:
    norm_array = np.append(norm_array,(x - center)*(12/500))  
Iplus = icurve(norm_array,1)
Iminus = icurve(norm_array,-1)
ratio = Iminus/Iplus
    
R_arr = []
P_arr = []
for x in range(0,5000):
    P = random.uniform(0,1)
    r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
    Iminus = icurve(norm_array,-1)
    array = r*Iminus
    array_flipped = np.flip(array)
    element_1 = array_flipped+Iminus
    sum_array = np.sum(array_flipped)*(12/500)
    element_2 = 1/sum_array
    element_3 = P
    noise = np.random.normal(0,1)
    result = element_1*element_2*element_3 + noise
    R_arr.append(result)
    df = pd.DataFrame(R_arr)
    P_arr.append(P)
df['P'] = P_arr
df.to_csv('Sample_Data_5k_with_noise.csv')