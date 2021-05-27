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
	model.add(Dense(512, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dropout(0.35))
	#model.add(Dense(1024, kernel_initializer='he_uniform', activation='relu'))
	#model.add(Dropout(0.25))
	#model.add(Dense(256, kernel_initializer='he_uniform', activation='relu'))
	#model.add(Dropout(0.05))
	model.add(Dense(n_outputs, kernel_initializer='he_uniform'))
	model.compile(loss='mae', optimizer='adam')
	return model

# load dataset
data = np.loadtxt("pseudo_data_3c.txt")
X = data[:,[0,1,2]]

y = data[:,[3,5]]
yerr = data[:,[4,6]]
F1_rep = np.random.normal(y[:,[0]], yerr[:,[0]])
F2_rep = np.random.normal(y[:,[1]], yerr[:,[1]])

yrep = np.column_stack((F1_rep,F2_rep))
n_inputs, n_outputs = X.shape[1], y.shape[1]
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(X, yrep, verbose=0, epochs=2500)
# make a prediction for new data
for x in range(400):
 z = x*0.0025 + 0.00125
 xvar = x*0.0025 + 0.00125
 #row = [0.6,0.3,z]
 row = [xvar,0.4,0.25]
 newX = asarray([row])
 yhat = model.predict(newX)
 print('%s' % yhat[0]) 
