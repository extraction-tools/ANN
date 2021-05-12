# use mlp for prediction on multi-output regression
#from numpy import asarray
import numpy as np
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(2048, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(1024, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dropout(0.25))
	#model.add(Dense(256, kernel_initializer='he_uniform', activation='relu'))
	#model.add(Dropout(0.05))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	model.compile(loss='mae', optimizer='adam')
	return model

# load dataset
data = np.loadtxt("data.txt")
#X, y = get_dataset()
X = data[:,[0,1,2,3]]
y = data[:,[13,15,17]]
yerr = data[:,[14,16,18]]
ReH_rep = np.random.normal(y[:,[0]], yerr[:,[0]])
ReE_rep = np.random.normal(y[:,[1]], yerr[:,[1]])
ReHT_rep = np.random.normal(y[:,[2]], yerr[:,[2]])
yrep = np.column_stack((ReH_rep,ReE_rep,ReHT_rep))
n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, yrep, verbose=0, epochs=4000)
# make a prediction for new data
for x in range(400):
 t = -0.5 + x*0.001 + 0.0005
 Q = 0.5 + x*0.1 + 0.05
 k = 2 + x*0.02 + 0.01
 row = [4.75,2.036,0.416,t]
 newX = asarray([row])
 yhat = model.predict(newX)
 print('%s' % yhat[0])
