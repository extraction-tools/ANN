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
import sklearn.preprocessing 
import keras_tuner

df = pd.read_csv('/project/ptgroup/Devin/Neural_Network/Sample_Data_v2/Sample_Data_TE_v1/Sample_Data_500k.csv')

def split_data(X, y, split=0.75):
    temp = np.random.choice(list(range(len(y))), size=int(len(y)*split), replace=False)
    
    tst_X = pd.DataFrame.from_dict({k: v[temp] for k,v in X.items()})
    trn_X = pd.DataFrame.from_dict({k: v.drop(temp) for k,v in X.items()})
    
    tst_y = y[temp]
    trn_y = y.drop(temp)
    
#     tst_err = err[tstidxs]
#     trn_err = np.delete(err, tstidxs)
    
    return trn_X, tst_X, trn_y, tst_y

y = df['P']
x = df.drop(['P','SNR'],axis=1)
train_X, test_X, train_y, test_y = split_data(x,y)
# train_X = sklearn.preprocessing.StandardScaler().fit_transform(train_X)
# train_y = sklearn.preprocessing.StandardScaler().fit_transform(train_y.reshape(-1,1))
# test_X = sklearn.preprocessing.StandardScaler().fit_transform(test_X)
# test_y = sklearn.preprocessing.StandardScaler().fit_transform(test_y.reshape(-1,1))

# def create_model():
#     input = tf.keras.Input(shape=(500))
#     batch = tf.keras.layers.BatchNormalization()(input)
#     layer1 = tf.keras.layers.Dense(500, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch)
#     batch2 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer1)
#     layer2 = tf.keras.layers.Dense(400, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch2)
#     batch3 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer2)
#     layer3 = tf.keras.layers.Dense(250,activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch3)
#     batch4 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer3)
#     layer4 = tf.keras.layers.Dense(50, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch4)
#     batch5 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer4)
#     layer5 = tf.keras.layers.Dense(5, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch5)
#     batch6 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.5), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer5)
#     output = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch6)
#     model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
#     model.compile(optimizer = tf.keras.optimizers.Adam(.001),
#                         loss = tf.keras.losses.MeanSquaredError())
#     return model

def create_model():
    input = tf.keras.Input(shape=(500))
    batch = tf.keras.layers.BatchNormalization()(input)
    layer1 = tf.keras.layers.Dense(250, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch)
    batch2 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer1)
    layer2 = tf.keras.layers.Dense(100, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch2)
    batch3 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer2)
    layer3 = tf.keras.layers.Dense(75,activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch3)
    batch4 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer3)
    layer4 = tf.keras.layers.Dense(50, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch4)
    batch5 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer4)
    layer5 = tf.keras.layers.Dense(5, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch5)
    batch6 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer5)
    output = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch6)
    model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
    model.compile(optimizer = tf.keras.optimizers.Adam(.005),
                        loss = tf.keras.losses.MeanSquaredError())
    return model

# def create_model():
#     input = tf.keras.Input(shape=(500))
#     batch = tf.keras.layers.BatchNormalization()(input)
#     layer1 = tf.keras.layers.Dense(100, activation = 'relu6', kernel_regularizer=regularizers.L1L2(10**(-14)),activity_regularizer=regularizers.L1L2(10**(-14)))(batch)
#     layer2 = tf.keras.layers.Dense(90, activation = 'relu6', kernel_regularizer=regularizers.L1L2(10**(-14)),activity_regularizer=regularizers.L1L2(10**(-14)))(layer1)
#     # layer3 = tf.keras.layers.Dense(70, activation = 'relu6', kernel_regularizer=regularizers.L1L2(10**(-14)),activity_regularizer=regularizers.L1L2(10**(-14)))(layer2)
#     layer4 = tf.keras.layers.Dense(50,activation = 'relu6', kernel_regularizer=regularizers.L1L2(10**(-14)),activity_regularizer=regularizers.L1L2(10**(-14)))(layer2)
#     # layer5 = tf.keras.layers.Dense(30,activation = 'relu6', kernel_regularizer=regularizers.L1L2(10**(-14)),activity_regularizer=regularizers.L1L2(10**(-14)))(layer4)
#     layer6 = tf.keras.layers.Dense(25, activation = 'relu6', kernel_regularizer=regularizers.L1L2(10**(-14)),activity_regularizer=regularizers.L1L2(10**(-14)))(layer4)
#     # layer7 = tf.keras.layers.Dense(10, activation = 'relu6', kernel_regularizer=regularizers.L1L2(10**(-14)),activity_regularizer=regularizers.L1L2(10**(-14)))(layer6)
#     output = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L1L2(10**(-14)),activity_regularizer=regularizers.L1L2(10**(-14)))(layer6)
#     model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
#     model.compile(optimizer = tf.keras.optimizers.Adam(.001),
#                         loss = tf.keras.losses.MeanSquaredError())
#     return model

# def create_model():
#     input = tf.keras.Input(shape=(500))
#     batch = tf.keras.layers.BatchNormalization()(input)
#     layer1 = tf.keras.layers.Dense(400, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch)
#     dropout1 = tf.keras.layers.Dropout(.2)(layer1)
#     layer2 = tf.keras.layers.Dense(350, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(dropout1)
#     dropout2 = tf.keras.layers.Dropout(.2)(layer1)
#     layer3 = tf.keras.layers.Dense(200, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(dropout2)
#     dropout3 = tf.keras.layers.Dropout(.2)(layer2)
#     layer4 = tf.keras.layers.Dense(100,activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(dropout3)
#     dropout4 = tf.keras.layers.Dropout(.2)(layer3)
#     layer5 = tf.keras.layers.Dense(30,activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(dropout4)
#     dropout5 = tf.keras.layers.Dropout(.2)(layer4)
#     layer6 = tf.keras.layers.Dense(25, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(dropout5)
#     dropout6 = tf.keras.layers.Dropout(.2)(layer5)
#     layer7 = tf.keras.layers.Dense(10, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(dropout6)
#     dropout7 = tf.keras.layers.Dropout(.2)(layer6)
#     output = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(dropout7)
#     model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
#     model.compile(optimizer = tf.keras.optimizers.Adam(.001),
#                         loss = tf.keras.losses.MeanSquaredError())
#     return model

# def create_model():
#     input = tf.keras.Input(shape=(500))
#     batch = tf.keras.layers.BatchNormalization()(input)
#     layer1 = tf.keras.layers.Dense(400, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch)
#     batch2 = tf.keras.layers.BatchNormalization()(layer1)
#     layer2 = tf.keras.layers.Dense(350, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch2)
#     batch3 = tf.keras.layers.BatchNormalization()(layer2)
#     layer3 = tf.keras.layers.Dense(200, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch3)
#     batch4 = tf.keras.layers.BatchNormalization()(layer3)
#     layer4 = tf.keras.layers.Dense(100,activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch4)
#     batch5 = tf.keras.layers.BatchNormalization()(layer4)
#     layer5 = tf.keras.layers.Dense(30,activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch5)
#     batch6 = tf.keras.layers.BatchNormalization()(layer5)
#     layer6 = tf.keras.layers.Dense(25, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch6)
#     batch7 = tf.keras.layers.BatchNormalization()(layer6)
#     layer7 = tf.keras.layers.Dense(10, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch7)
#     batch8 = tf.keras.layers.BatchNormalization()(layer7)
#     output = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(batch8)
#     model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
#     model.compile(optimizer = tf.keras.optimizers.Adam(.001),
#                         loss = tf.keras.losses.MeanSquaredError())
#     return model

# def create_model():
#     input = tf.keras.Input(shape=(500))
#     batch = tf.keras.layers.BatchNormalization()(input)
#     layer1 = tf.keras.layers.Dense(50, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-17)),activity_regularizer=regularizers.L2(10**(-17)))(batch)
#     layer2 = tf.keras.layers.Dense(25, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-17)),activity_regularizer=regularizers.L2(10**(-17)))(layer1)
#     # layer3 = tf.keras.layers.Dense(250,activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(layer2)
#     # layer4 = tf.keras.layers.Dense(50, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-14)),activity_regularizer=regularizers.L2(10**(-14)))(layer3)
#     layer5 = tf.keras.layers.Dense(5, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-17)),activity_regularizer=regularizers.L2(10**(-17)))(layer2)
#     output = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(10**(-17)),activity_regularizer=regularizers.L2(10**(-17)))(layer5)
#     model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
#     model.compile(optimizer = tf.keras.optimizers.Adam(.001),
#                         loss = tf.keras.losses.MeanSquaredError())
#     return model
# ##Best Model So Far

model_test = create_model()
model_test.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, mode='auto')

csv_logger = CSVLogger('Trained_Models_v12/log_500K_v8.csv', append=True, separator=';')
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', patience = 15)
fitted_data = model_test.fit(train_X,train_y, validation_data = (test_X, test_y), epochs = 300,callbacks=[csv_logger, earlystop],batch_size = 1024)

model_test.save('Trained_Models_v12/trained_model_500K_v8.h5', save_format='h5')

plt.plot(fitted_data.history['loss'])
plt.plot(fitted_data.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val_loss'], loc='upper left')
plt.savefig('Trained_Models_v12/trained_model_loss_500K_v8.jpeg')