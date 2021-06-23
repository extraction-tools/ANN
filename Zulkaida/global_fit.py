# use mlp for prediction on multi-output regression
#from numpy import asarray
from tensorflow import keras
import numpy as np
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=10000,
    decay_rate=0.9)

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(1024, input_dim=n_inputs, kernel_initializer='he_uniform', activation='tanh'))
	#model.add(Dropout(0.2))
	#model.add(Dense(100, kernel_initializer='he_uniform', activation='relu'))
	#model.add(Dropout(0.25))
	model.add(Dense(104, kernel_initializer='he_uniform', activation='relu'))
	#model.add(Dropout(0.05))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	opt = keras.optimizers.Adam(learning_rate=lr_schedule)
	model.compile(loss='mae', optimizer=opt)
	return model

#early stopping
keras_callbacks   = [
      EarlyStopping(monitor='loss', patience=20, mode='min', min_delta=0.000001)
      
]

# load dataset
data = np.loadtxt("lilietfit.txt")
#X, y = get_dataset()
X = data[:,[0,1,2,3]]
y = data[:,[4,6,8]]
yerr = data[:,[5,7,9]]
ReH_rep = np.random.normal(y[:,[0]], yerr[:,[0]])
ReE_rep = np.random.normal(y[:,[1]], yerr[:,[1]])
ReHT_rep = np.random.normal(y[:,[2]], yerr[:,[2]])
yrep = np.column_stack((ReH_rep,ReE_rep,ReHT_rep))
n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, yrep, verbose=0, epochs=6000, callbacks = keras_callbacks)
# make a prediction for new data
for x in range(400):
 t = -0.5 + x*0.001 + 0.0005
 Q = 0.5 + x*0.1 + 0.05
 k = 2 + x*0.02 + 0.01
 # row is the kinemtics that we want to predict
 #row = [4.75,2.036,0.416,t]
 row = [5.75,2.72254,0.47467,t]
 newX = asarray([row])
 yhat = model.predict(newX)
 truth_ReH = 5. * t*t + 2.* 0.47467*0.7467
 truth_ReE = -1.5 * 0.47467*0.47467 + 4.5 * t
 truth_ReHT = 4.5 * 0.47567 - 5.5 * t
 #print('%s %.3f %.3f %.3f' % (yhat[0],truth_ReH, truth_ReE, truth_ReHT))
 print('%s' % yhat[0])
