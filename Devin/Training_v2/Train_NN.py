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
from keras.callbacks import CSVLogger

df = pd.read_csv('/project/ptgroup/Devin/Neural_Network/Sample_Data/Sample_Data_1M_0p/Sample_Data_1M.csv')

def split_data(X, y, split=0.90):
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

def create_model():
    input = tf.keras.Input(shape=(500))
    layer1 = tf.keras.layers.Dense(500, activation = 'relu6',kernel_regularizer=regularizers.L2(10**(-15)),activity_regularizer=regularizers.L2(10**(-15)))(input)
    # layer2 = tf.keras.layers.Dropout(0.1)(layer1)
    layer3 = tf.keras.layers.Dense(400, activation = 'relu6',kernel_regularizer=regularizers.L2(10**(-15)),activity_regularizer=regularizers.L2(10**(-15)))(layer1)
    layer4 = tf.keras.layers.Dense(120, activation = 'relu6',kernel_regularizer=regularizers.L2(10**(-15)),activity_regularizer=regularizers.L2(10**(-15)))(layer3)
    # layer4 = tf.keras.layers.Dropout(0.1)(layer3)
    # layer5 = tf.keras.layers.Dense(256, activation = 'relu',kernel_regularizer=regularizers.L2(10**(-12)),activity_regularizer=regularizers.L2(10**(-12)))(layer4)
    # layer6 = tf.keras.layers.Dropout(0.0)(layer5)
    # layer7 = tf.keras.layers.Dense(256, activation = 'relu',kernel_regularizer=regularizers.L2(10**(-12)),activity_regularizer=regularizers.L2(10**(-12)))(layer6)
    # layer8 = tf.keras.layers.Dropout(0.0)(layer7)
    layer9 = tf.keras.layers.Dense(70, activation = 'relu6',kernel_regularizer=regularizers.L2(10**(-15)),activity_regularizer=regularizers.L2(10**(-15)))(layer4)
    layer10 = tf.keras.layers.Dense(25, activation = 'relu6',kernel_regularizer=regularizers.L2(10**(-15)),activity_regularizer=regularizers.L2(10**(-15)))(layer9)
    output = tf.keras.layers.Dense(1,activation='sigmoid',kernel_regularizer=regularizers.L2(10**(-15)),activity_regularizer=regularizers.L2(10**(-15)))(layer10)
    model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
    model.compile(optimizer = tf.keras.optimizers.Adam(.005),
                        loss = tf.keras.losses.MeanSquaredError())
    return model
# Most Accurate Model So Far

model_test = create_model()
model_test.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, mode='auto')

csv_logger = CSVLogger('log_v52.csv', append=True, separator=';')
# earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
fitted_data = model_test.fit(train_X,train_y, validation_data = (test_X, test_y), epochs = 150,callbacks=[reduce_lr,csv_logger],batch_size=len(train_X))

model_test.save('Trained_Models_v2/trained_model_1M_redux_v52.h5', save_format='h5')

plt.plot(fitted_data.history['loss'])
plt.plot(fitted_data.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val_loss'], loc='upper left')
plt.savefig('/project/ptgroup/Devin/Neural_Network/Sample_Data/trained_model_loss_1M_redux_v52.jpeg')