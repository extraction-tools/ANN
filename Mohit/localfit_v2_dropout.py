import numpy as np
import pandas as pd
from BHDVCS_tf import *
import tensorflow as tf

import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

import os

data_number = '3'  # the data file to use

df = pd.read_csv("test_data/BKM_pseudodata" +
                 data_number + ".csv", dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

data = DvcsData(df)


kinematics = tf.keras.Input(shape=(4))
x1 = tf.keras.layers.Dense(100, activation="tanh")(kinematics)
d1 = tf.keras.layers.Dropout(0)(x1)
x2 = tf.keras.layers.Dense(100, activation="tanh")(d1)
d2 = tf.keras.layers.Dropout(0)(x2)
outputs = tf.keras.layers.Dense(4, activation="linear")(d2)
noncffInputs = tf.keras.Input(shape=(7))
#### phi, kin1, kin2, kin3, kin4, F1, F2 ####
total_FInputs = tf.keras.layers.concatenate([noncffInputs, outputs])
TotalF = TotalFLayer()(total_FInputs)

tfModel = tf.keras.Model(
    inputs=[kinematics, noncffInputs], outputs=TotalF, name="tfmodel")
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=0.0000005, patience=25)

tfModel.compile(
    optimizer=tf.keras.optimizers.Adam(.0085),
    loss=tf.keras.losses.MeanSquaredError()
)

Wsave = tfModel.get_weights()


def get_total_error(experimental_values, expected_values):  # gets the mean absolute error
  experimental_values, expected_values = list(
      experimental_values), list(expected_values)
  tot = 0
  for i, j in zip(experimental_values, expected_values):
    tot += abs(float(j) - float(i))
  return tot / len(experimental_values)


# gets the maximum residual
def get_max_residual(x_values, experimental_values, expected_values):
  x_values, experimental_values, expected_values = list(
      x_values), list(experimental_values), list(expected_values)
  maximum = 0
  for n, (i, j) in enumerate(zip(experimental_values, expected_values)):
    residual = abs(float(j) - float(i))
    if residual > maximum:
      maximum = residual
  return maximum


def get_rms(experimental_values, expected_values):  # normalized root mean square error
  experimental_values, expected_values = list(
      experimental_values), list(expected_values)
  tot = 0
  for i, j in zip(experimental_values, expected_values):
    tot += (float(j) - float(i))**2

  tot /= len(experimental_values)
  tot = np.sqrt(tot)
  tot /= (np.mean(expected_values))
  return tot


def F2VsPhi_noPlot(dataframe, SetNum, xdat, cffs):
  f = BHDVCStf().curve_fit
  TempFvalSilces = dataframe[dataframe["#Set"] == SetNum]
  TempFvals = TempFvalSilces["F"]
  temp_phi = TempFvalSilces["phi_x"]

  calculated_points = f(xdat, cffs)

  return (calculated_points, get_total_error(calculated_points, TempFvals),
          get_max_residual(temp_phi, calculated_points, TempFvals),
          get_rms(calculated_points, TempFvals))


################################################# FINDING THE BEST COMBINATION OF EPOCH AND BATCH #######################################


total_errors = {}
total_residuals = {}
total_rms_vals = {}
cffs_record = {}  # records all of the cffs
F_vals = {}


testnum = int(df['#Set'].max())
skip = 50  # samples a series of different sets

for epoch in np.arange(10, 1000, 50):  # parse the upper region less thoroughly
  for batch in np.arange(1, 11):
    for i in np.arange(0, testnum, skip):
      for d_rate in np.arange(0, 0.61, 0.2):
        for trial in np.arange(0, 10):

          for n, layer in enumerate(tfModel.layers):
            if 'dropout' in layer.name:
              # changes the dropout rate for the model
              tfModel.layers[n].rate = d_rate

          tfModel.compile(
              optimizer=tf.keras.optimizers.Adam(.0085),
              loss=tf.keras.losses.MeanSquaredError()
          )

          tfModel.set_weights(Wsave)  # resets the model
          setI = data.getSet(i, itemsInSet=45)

          # one replica of samples from F vals
          tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(),
                      epochs=epoch, verbose=0, batch_size=batch,
                      callbacks=[early_stopping_callback], validation_split=0.2)

          # 2 hidden layers and 2 dropout layers for numHL parameter
          cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=4)

          new_xdat = np.transpose(setI.XnoCFF.to_numpy(dtype=np.float32))

          # Avoid recalculating F-values from cffs when that is what the model is predicting already

          F, total_error, max_residual, total_rms = F2VsPhi_noPlot(
              df, i + 1, new_xdat, cffs
          )  # runs the version without plotting to save time

          # the inputs to place in the dictionary
          selection_key = (epoch, batch, d_rate, trial, i)
          F_vals[selection_key] = np.array(F)
          cffs_record[selection_key] = np.array(cffs)
          total_errors[selection_key] = total_error
          total_residuals[selection_key] = max_residual
          total_rms_vals[selection_key] = total_rms


# for epoch in np.arange(1000, 7001, 500):  # parse the upper region less thoroughly
#   # 46 is greater than the 45 we need, but it will floor to 45
#   for batch in np.arange(1, 47, 5):
#     for i in np.arange(0, testnum, skip):
#       for d_rate in np.arange(0, 0.61, 0.2):

#         for n, layer in enumerate(tfModel.layers):
#           if 'dropout' in layer.name:
#             # changes the dropout rate for the model
#             tfModel.layers[n].rate = d_rate

#         tfModel.compile(
#             optimizer=tf.keras.optimizers.Adam(.0085),
#             loss=tf.keras.losses.MeanSquaredError()
#         )

#         tfModel.set_weights(Wsave)  # resets the model
#         setI = data.getSet(i, itemsInSet=45)

#         tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(),  # one replica of samples from F vals
#                     epochs=epoch, verbose=0, batch_size=batch, callbacks=[early_stopping_callback], validation_split=0.2)

#         cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=4)

#         new_xdat = np.transpose(setI.XnoCFF.to_numpy(dtype=np.float32))

#         # Avoid recalculating F-values from cffs when that is what the model is predicting already

#         F, total_error, max_residual, total_rms = F2VsPhi_noPlot(
#             df, i + 1, new_xdat, cffs
#         )  # runs the version without plotting to save time

#         selection_key = (epoch, batch, d_rate, i)
#         F_vals[selection_key] = np.array(F)
#         cffs_record[selection_key] = np.array(cffs)
#         total_errors[selection_key] = total_error
#         total_residuals[selection_key] = max_residual
#         total_rms_vals[selection_key] = total_rms

base_cols = ["Epoch", "Batch", "Dropout Rate", "Trial", "Set"]

F_vals = pd.Series(F_vals).reset_index()
F_vals.columns = base_cols + ["Calculated Points"]

cffs_record = pd.Series(cffs_record).reset_index()
cffs_record.columns = base_cols + ["Calculated CFFs"]

total_errors = pd.Series(total_errors).reset_index()
total_errors.columns = base_cols + ["MAE"]

total_residuals = pd.Series(total_residuals).reset_index()
total_residuals.columns = base_cols + ["Max Residual"]

total_rms_vals = pd.Series(total_rms_vals).reset_index()
total_rms_vals.columns = base_cols + ["NRMSE"]

total_metrics = F_vals.merge(cffs_record).merge(
    total_errors).merge(total_residuals).merge(total_rms_vals)

total_metrics['Set'] += 1

base_filestr = 'metrics' + data_number + '_dropout'
final_str = base_filestr
filestr_num = 1
while os.path.exists(final_str + '.csv'):
  final_str = base_filestr + '_' + str(filestr_num)
  filestr_num += 1

total_metrics.to_csv(final_str + '.csv')  # ensures data is saved