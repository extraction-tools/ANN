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
import keras_tuner as kt
from keras.models import Sequential

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
#     layer1 = tf.keras.layers.Dense(250, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch)
#     batch2 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer1)
#     layer2 = tf.keras.layers.Dense(100, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch2)
#     batch3 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer2)
#     layer3 = tf.keras.layers.Dense(75,activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch3)
#     batch4 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer3)
#     layer4 = tf.keras.layers.Dense(50, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch4)
#     batch5 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer4)
#     layer5 = tf.keras.layers.Dense(5, activation = 'relu6', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch5)
#     batch6 = tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9))(layer5)
#     output = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25)))(batch6)
#     model = tf.keras.Model(inputs=[input], outputs=output, name="testmodel")
#     model.compile(optimizer = tf.keras.optimizers.Adam(.005),
#                         loss = tf.keras.losses.MeanSquaredError())
#     return model
    
def model(hp):
    model = Sequential()
    model.add(tf.keras.Input(shape=(500)))
    model.add(tf.keras.layers.BatchNormalization())
    hp_units = hp.Int('units', min_value=1, max_value=500, step=50)
    model.add(tf.keras.layers.Dense(units=hp_units, activation = 'relu', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25))))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=hp_units, activation = 'relu', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25))))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=hp_units,activation = 'relu', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25))))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=hp_units, activation = 'relu', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25))))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(units=hp_units, activation = 'relu', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25))))
    model.add(tf.keras.layers.BatchNormalization(momentum=0.95, epsilon=0.005, beta_initializer=tf.keras.initializers.RandomNormal(0.0, 0.05), gamma_initializer=tf.keras.initializers.Constant(value=0.9)))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_regularizer=regularizers.L2(10**(-25)),activity_regularizer=regularizers.L2(10**(-25))))
    hp_learning_rate = hp.Choice('learning_rate',values = [1e-3,1e-4])
    opt = tf.keras.optimizers.Adam(learning_rate = hp_learning_rate)
    model.compile(optimizer = opt, loss = 'mean_squared_error',metrics=['mean_squared_error'])
    return model
    
    
tuner = kt.Hyperband(model,
                     objective='val_loss',
                     max_epochs=100,
                     factor=3,
                     project_name='NMR')

# model_test = create_model()
# model_test.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, mode='auto')

csv_logger = CSVLogger('Trained_Models_v12/log_500K_v12.csv', append=True, separator=';')
earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode = 'min', patience = 5)

tuner.search(train_X, train_y,validation_data = (test_X, test_y), epochs=50, validation_split=0.7, callbacks=[earlystop])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

fitted_data = model.fit(train_X,train_y, validation_data = (test_X, test_y), epochs = 300,callbacks=[csv_logger, earlystop],batch_size = 124)

model.save('Trained_Models_v12/trained_model_500K_v12.h5', save_format='h5')

plt.plot(fitted_data.history['loss'])
plt.plot(fitted_data.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val_loss'], loc='upper left')
plt.savefig('Trained_Models_v12/trained_model_loss_500K_v12.jpeg')