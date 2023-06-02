#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


import numpy as np
import matplotlib.pyplot as mp
from matplotlib.ticker import FormatStrFormatter
import matplotlib.font_manager as font_manager

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

minor_ticks_x = np.arange(-3,3.5,0.5)
minor_ticks_y = np.arange(0,0.6,0.05)
fig = mp.figure()
ax = fig.add_subplot(1, 1, 1)

axisFontSize = 12
titleFontSize = 12
legendFontSize = 12

font = font_manager.FontProperties(family='serif',size=legendFontSize)
mp.plot(xvals, yvals+0.5*yvals2, "r",linewidth = 4)
# mp.plot(xvals, yvals, "g",linewidth = 4)
# mp.plot(xvals, 0.5*yvals2, "b",linewidth = 4)
mp.xlabel('R', fontsize=axisFontSize)
mp.ylabel('Intensity [$C_{E}$ mV]', fontsize=axisFontSize)
mp.grid(True, which='both', axis='both', linewidth=2)
mp.xticks(minor_ticks_x, fontsize=axisFontSize)
mp.yticks(minor_ticks_y, fontsize=axisFontSize)
for label in ax.xaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
for label in ax.yaxis.get_ticklabels()[1::2]:
    label.set_visible(False)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
mp.xlim(-3, 3)
ax.set_axisbelow(True)

fig = mp.gcf()
# fig.set_size_inches(16, 16)
plt.show()


# In[3]:


P = .15
r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
print(r)


# In[4]:


center = 250
length = range(501)
norm_array = []
for x in length:
    norm_array = np.append(norm_array,(x - center)*(12/500))  
Iplus = icurve(norm_array,1)
Iminus = icurve(norm_array,-1)
ratio = Iminus/Iplus
polar = []
for i in length:
    polar = np.append(polar,.8)


# In[5]:


### Polarization ###
array = r*Iminus
array_flipped = np.flip(array)
element_1 = array_flipped+Iminus
sum_array = np.sum(array_flipped)*(12/500)
element_2 = 1/sum_array
element_3 = P
result = element_1*element_2*element_3
plt.plot(norm_array,result)
plt.ylim(0,1)
plt.xlim(-3,3)
df = pd.DataFrame({'R':result})
# df_T = df.T
# df_T.to_csv("Sample_Data.csv",index=False)


# In[6]:


# R_arr = []
# P_arr = []
# for x in range(0,100):
#     P = random.uniform(0,1)
#     r = (np.sqrt(4-3*P**(2))+P)/(2-2*P)
#     Iminus = icurve(norm_array,-1)
#     array = r*Iminus
#     array_flipped = np.flip(array)
#     element_1 = array_flipped+Iminus
#     sum_array = np.sum(array_flipped)*(12/500)
#     element_2 = 1/sum_array
#     element_3 = P
#     result = element_1*element_2*element_3
#     R_arr.append(result)
#     df = pd.DataFrame(R_arr)
#     P_arr.append(P)
# df['P'] = P_arr
# df


# In[6]:


R_arr = []
P_arr = []
for x in range(0,500):
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
    result = element_1*element_2*element_3
    R_arr.append(result)
    df = pd.DataFrame(R_arr)
    P_arr.append(P)
df['P'] = P_arr
df


# In[37]:


df.to_csv("Sample_Data.csv")


# In[26]:


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
train_X, test_X, train_y, test_y = split_data(x,y)


# In[27]:


def create_model():
    input = tf.keras.Input(shape=(501))
#     noise = tf.keras.layers.GaussianNoise(0.1)(input)
    layer1 = tf.keras.layers.Dense(256, activation = 'relu',activity_regularizer=regularizers.L2(10**(-10)))(input)
#     noise = tf.keras.layers.GaussianNoise(0.1)(layer1)
#     drop1 = tf.keras.layers.Dropout(.15)(layer1)
    layer2 = tf.keras.layers.Dense(256, activation = 'relu',activity_regularizer=regularizers.L2(10**(-10)))(layer1)
#     drop2 = tf.keras.layers.Dropout(.15)(layer2)
    layer3 = tf.keras.layers.Dense(256, activation = 'relu',activity_regularizer=regularizers.L2(10**(-10)))(layer2)
    output = tf.keras.layers.Dense(1,activation='sigmoid',activity_regularizer=regularizers.L2(10**(-10)))(layer3)
    model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
    model.compile(optimizer = tf.keras.optimizers.Adam(.005),
                        loss = tf.keras.losses.MeanSquaredError())
    return model


# In[28]:


model_test = create_model()
model_test.summary()


# In[29]:


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, mode='auto')


# In[30]:


fitted_data = model_test.fit(train_X,train_y, validation_data = (test_X, test_y), epochs = 100,callbacks=[reduce_lr],batch_size=30)


# In[31]:


model_test.save('trained_model_Noise_v2.h5', save_format='h5')


# In[32]:


plt.plot(fitted_data.history['loss'])
# plt.plot(fitted_data.history['val_loss'])
# plt.title('model loss w/ 1000 epochs, learning rate of 0.000002')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
# plt.savefig('trained_model_1.jpeg')
plt.show()


# In[33]:


testmodel = tf.keras.models.load_model('trained_model_Noise_v2.h5')


# In[34]:


X = train_X
Y = testmodel.predict(X)
P = np.array(P_arr)
Y = Y.reshape((500,))
result = pd.DataFrame(Y)
result = result.rename(columns={df.columns[0]:'P'})
Accuracy = ((P_arr - Y)/P_arr)
# result['Err'] = pd.DataFrame(Accuracy).abs()
result['P_True'] = P_arr
# plt.errorbar(np.arange(len(result['P'])),
#              result['P'],
#              yerr=result['Err'],label="Pre",fmt='.')

# plt.plot(np.arange(len(result['P'])),result['P'],'.')
# plt.xlabel('Data Point ID')
# plt.ylabel('Predicted Polarization')
plt.plot(np.arange(len(result['P_True'])),result['P_True'],'.',label = 'True')
plt.plot(np.arange(len(result['P'])),result['P'],'.')
plt.xlabel('Data Point ID')
plt.ylabel('Predicted Polarization')
plt.legend(['Pre','True'])
# plt.savefig("NN_Example.jpeg")


# In[35]:


result['P'].plot.bar(figsize=(100,20),alpha=1,color='red',label='Predicted')
result['P_True'].plot.bar(figsize=(100,20),alpha=1,color='black',label='True')
plt.xlabel('Data Point',fontsize=64)
plt.ylabel('Polarization',fontsize=64)
# plt.xticks(fontsize=32)


# In[36]:


# df_mask = result['Err'] < 1
# filt = result[df_mask]
filt = result
upper_lim = 100
lower_lim = 0
plt.plot(np.arange(len(filt['P'][lower_lim:upper_lim])),filt['P'][lower_lim:upper_lim],'.',label="Pre")
plt.xlabel('Data Point ID')
plt.ylabel('Predicted Polarization')
plt.plot(np.arange(len(result['P_True'][lower_lim:upper_lim])) + .5,result['P_True'][lower_lim:upper_lim],'.',label = 'True')
plt.legend(['True','Pre'])
plt.title('Data Point ID: 0 - 100')
# plt.savefig('Plots/NN_Example_1.jpeg')


# In[ ]:




