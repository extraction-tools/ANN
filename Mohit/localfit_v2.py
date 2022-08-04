import numpy as np
import pandas as pd
from BHDVCS_tf import *
import tensorflow as tf

import matplotlib.pyplot as plt

import sys
from scipy.stats import chisquare

import os

data_number = '2'  # the data file to use

df = pd.read_csv("test_data/BKM_pseudodata" +
                 data_number + ".csv", dtype=np.float64)
df = df.rename(columns={"sigmaF": "errF"})

data = DvcsData(df)


kinematics = tf.keras.Input(shape=(4))
x1 = tf.keras.layers.Dense(100, activation="tanh")(kinematics)
x2 = tf.keras.layers.Dense(100, activation="tanh")(x1)
outputs = tf.keras.layers.Dense(4, activation="linear")(x2)
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
skip = 100  # samples a series of different sets

for epoch in np.arange(10, 1000, 50):  # parse the upper region less thoroughly
  # 46 is greater than the 45 we need, but it will floor to 45
  for batch in np.arange(1, 11, 2):
    for i in np.arange(0, testnum, skip):
      tfModel.set_weights(Wsave)  # resets the model
      setI = data.getSet(i, itemsInSet=45)

      tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(),  # one replica of samples from F vals
                  epochs=epoch, verbose=0, batch_size=batch, callbacks=[early_stopping_callback], validation_split=0.2)

      cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

      new_xdat = np.transpose(setI.XnoCFF.to_numpy(dtype=np.float32))

      # Avoid recalculating F-values from cffs when that is what the model is predicting already

      F, total_error, max_residual, total_rms = F2VsPhi_noPlot(
          df, i + 1, new_xdat, cffs
      )  # runs the version without plotting to save time

      F_vals[(epoch, batch, i)] = np.array(F)
      cffs_record[(epoch, batch, i)] = np.array(cffs)
      total_errors[(epoch, batch, i)] = total_error
      total_residuals[(epoch, batch, i)] = max_residual
      total_rms_vals[(epoch, batch, i)] = total_rms


for epoch in np.arange(1000, 30001, 500):  # parse the upper region less thoroughly
  # 46 is greater than the 45 we need, but it will floor to 45
  for batch in np.arange(1, 47, 5):
    for i in np.arange(0, testnum, skip):
      tfModel.set_weights(Wsave)  # resets the model
      setI = data.getSet(i, itemsInSet=45)

      tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(),  # one replica of samples from F vals
                  epochs=epoch, verbose=0, batch_size=batch, callbacks=[early_stopping_callback], validation_split=0.2)

      cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

      new_xdat = np.transpose(setI.XnoCFF.to_numpy(dtype=np.float32))

      # Avoid recalculating F-values from cffs when that is what the model is predicting already

      F, total_error, max_residual, total_rms = F2VsPhi_noPlot(
          df, i + 1, new_xdat, cffs
      )  # runs the version without plotting to save time

      F_vals[(epoch, batch, i)] = np.array(F)
      cffs_record[(epoch, batch, i)] = np.array(cffs)
      total_errors[(epoch, batch, i)] = total_error
      total_residuals[(epoch, batch, i)] = max_residual
      total_rms_vals[(epoch, batch, i)] = total_rms

F_vals = pd.Series(F_vals).reset_index()
F_vals.columns = ["Epoch", "Batch", "Set", "Calculated Points"]

cffs_record = pd.Series(cffs_record).reset_index()
cffs_record.columns = ["Epoch", "Batch", "Set", "Calculated CFFs"]

total_errors = pd.Series(total_errors).reset_index()
total_errors.columns = ["Epoch", "Batch", "Set", "MAE"]

total_residuals = pd.Series(total_residuals).reset_index()
total_residuals.columns = ["Epoch", "Batch", "Set", "Max Residual"]

total_rms_vals = pd.Series(total_rms_vals).reset_index()
total_rms_vals.columns = ["Epoch", "Batch", "Set", "NRMSE"]

total_metrics = F_vals.merge(cffs_record).merge(
    total_errors).merge(total_residuals).merge(total_rms_vals)

base_filestr = 'metrics' + data_number
final_str = base_filestr
filestr_num = 1
while os.path.exists(final_str + '.csv'):
  final_str = base_filestr + '_' + str(filestr_num)
  filestr_num += 1

total_metrics.to_csv(final_str + '.csv')  # ensures data is saved

# best_combination_errors_df = pd.DataFrame.from_dict(best_combination_errors, orient="index", columns=['epoch', 'batch', 'rms'])
# best_combination_errors_df.to_csv("best_combination_errors.csv")
# best_combination_residual_df = pd.DataFrame.from_dict(best_combination_residual, orient="index", columns=['epoch', 'batch', 'rms'])
# best_combination_residual_df.to_csv("best_combination_residual.csv")
# best_combination_rms_df = pd.DataFrame.from_dict(best_combination_rms, orient="index", columns=['epoch', 'batch', 'rms'])
# best_combination_rms_df.to_csv("best_combination_rms.csv")

# most_common = []
# for i,j,k in zip(best_combination_errors.values(), best_combination_residual.values(), best_combination_rms.values()):
#   most_common.append(i[:2])
#   most_common.append(j[:2])
#   most_common.append(k[:2])

# res_outcome = max(set(best_combination_residual.values()), key = list(best_combination_residual.values()).count)
# print("Just using residuals, the best epoch number is:", res_outcome[0], "with a batch size of", res_outcome[1])

# err_outcome = max(set(best_combination_errors.values()), key = list(best_combination_errors.values()).count)
# print("Just using error, the best epoch number is:", err_outcome[0], "with a batch size of", err_outcome[1])

# rms_outcome = max(set(best_combination_rms.values()), key = list(best_combination_rms.values()).count)
# print("Just using root mean square error, the best epoch number is:", rms_outcome[0], "with a batch size of", rms_outcome[1])

# final_outcome = max(set(most_common), key=most_common.count) #the final_outcome is a tuple of (epoch#, batch#)
# print("Using all 3 metrics, the best epoch number is: ", final_outcome[0], "with a batch size of", final_outcome[1])


# by_set = []
# by_set_metrics = []

# for i in range(0, testnum): #use the final outcome to have a final fit
#   by_metric_fits = []
#   for designator in ("err", "res", "rms", "final"):
#     setI = data.getSet(i, itemsInSet=45)

#     tfModel.set_weights(Wsave)
#     tfModel.fit([setI.Kinematics, setI.XnoCFF], setI.sampleY(), # one replica of samples from F vals
#                           epochs=eval(designator + "_outcome[0]"), verbose=0, batch_size=eval(designator + "_outcome[1]"), callbacks=[early_stopping_callback])

#     cffs = cffs_from_globalModel(tfModel, setI.Kinematics, numHL=2)

#     by_set.append([i, designator] + list(cffs))
#     by_set_metrics.append([i, designator] + list(F2VsPhi_noPlot(df.copy(),i+1,new_xdat,cffs)))

#     new_xdat = np.transpose(setI.XnoCFF.to_numpy(dtype=np.float32)) #NB: Could rewrite BHDVCS curve_fit to not require transposition

#     # Avoid recalculating F-values from cffs when that is what the model is predicting already
#     by_metric_fits.append(F2VsPhi_multiple_designations(df.copy(),i+1,new_xdat,cffs, designation = designator))

#   for n, fit in enumerate(by_metric_fits): #iterate over the fits that were just produced
#     if n == 0:
#       plt.errorbar(*fit[1],fmt='.',color='blue',label="Data") #this is the "real" data

#     plt.xlim(0,368)
#     TempFvals = fit[1][1]
#     temp_unit=(np.max(TempFvals)-np.min(TempFvals))/len(TempFvals)
#     plt.ylim(np.min(TempFvals)-temp_unit,np.max(TempFvals)+temp_unit)

#     grid_metric = ""
#     if fit[0] == "err":
#       grid_metric = "Mean Absolute Error"
#     elif fit[0] == "res":
#       grid_metric = "Maximum Residual"
#     elif fit[0] == "rms":
#       grid_metric = "NRMSE"
#     else:
#       grid_metric = "Most Common Hyperparameter Set"
#     plt.plot(*fit[2], label='fit using ' + grid_metric, linestyle='dashed')

#   plt.xticks(fontsize=15)
#   plt.yticks(fontsize=15)
#   plt.title("Local fits with data set #"+str(i),fontsize=20)
#   plt.xlabel("phi")
#   plt.ylabel("F")
#   plt.legend(loc='best', fontsize=10,handlelength=3)
#   file_name = "plot_set_number_"+str(i)+".png"
#   plt.savefig(file_name)
#   plt.clf()


# newdf = pd.DataFrame(by_set)
# newdf.to_csv('bySetCFFs.csv')

# newdf2 = pd.DataFrame(by_set_metrics, columns=["set", "designator", "abs_err", "max_res", "rms_err"])
# newdf2.to_csv("metrics.csv")
