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

df = pd.read_csv('/project/ptgroup/Devin/Neural_Network/Sample_Data/1M_Data/Sample_Data_1M.csv')

def split_data(X, y, split=0.1):
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

# def create_model():
#     input = tf.keras.Input(shape=(501))
#     layer1 = tf.keras.layers.Dense(1024, activation = 'relu',activity_regularizer=regularizers.L2(10**(-10)))(input)
#     layer2 = tf.keras.layers.Dropout(0.2)(layer1)
#     layer3 = tf.keras.layers.Dense(1024, activation = 'relu',activity_regularizer=regularizers.L2(10**(-10)))(layer2)
#     layer4 = tf.keras.layers.Dropout(0.2)(layer3)
#     layer5 = tf.keras.layers.Dense(1024, activation = 'relu',activity_regularizer=regularizers.L2(10**(-10)))(layer4)
#     output = tf.keras.layers.Dense(1,activation='sigmoid',activity_regularizer=regularizers.L2(10**(-10)))(layer5)
#     model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
#     model.compile(optimizer = tf.keras.optimizers.Adam(.005),
#                         loss = tf.keras.losses.MeanSquaredError())
#     return model

def create_model():
    input = tf.keras.Input(shape=(501))
    layer1 = tf.keras.layers.Dense(1024, activation = 'relu',kernel_regularizer=regularizers.L2(10**(-10)),activity_regularizer=regularizers.L2(10**(-10)))(input)
    layer2 = tf.keras.layers.Dropout(0.1)(layer1)
    layer3 = tf.keras.layers.Dense(1024, activation = 'relu',kernel_regularizer=regularizers.L2(10**(-10)),activity_regularizer=regularizers.L2(10**(-10)))(layer2)
    layer4 = tf.keras.layers.Dropout(0.1)(layer3)
    layer5 = tf.keras.layers.Dense(1024, activation = 'relu',kernel_regularizer=regularizers.L2(10**(-10)),activity_regularizer=regularizers.L2(10**(-10)))(layer4)
    output = tf.keras.layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.L2(10**(-10)),activity_regularizer=regularizers.L2(10**(-10)))(layer5)
    model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
    model.compile(optimizer = tf.keras.optimizers.Adam(.005),
                        loss = tf.keras.losses.MeanSquaredError())
    return model

model_test = create_model()
model_test.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, mode='auto')
                            
fitted_data = model_test.fit(train_X,train_y, validation_data = (test_X, test_y), epochs = 1000,callbacks=[reduce_lr],batch_size=1500)

model_test.save('Trained_Models/trained_model_1M.h5', save_format='h5')

plt.plot(fitted_data.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('/project/ptgroup/Devin/Neural_Network/Sample_Data/1M_Data/trained_model_loss_1M.jpeg')

plt.fig()
plt.plot(fitted_data.history['val_loss'])
plt.title('model val loss')
plt.ylabel('val loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('/project/ptgroup/Devin/Neural_Network/Sample_Data/1M_Data/trained_model_val_loss_100.jpeg')
