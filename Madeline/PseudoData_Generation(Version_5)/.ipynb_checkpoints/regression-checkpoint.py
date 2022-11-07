import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf # Version 2.9.1
from tensorflow import keras
from tensorflow.keras import layers
np.set_printoptions(precision=3, suppress=True)

column_names = ['count','Set','index','k','QQ','x_b','t','phi_x','F','sigmaF','F1','F2','ReH_true','ReE_true','ReHTilde_true','c0_true','Formalism']

dataset = [10, 20, 30, 40, 50, 60, 70, 80, 90]
fig, axs = plt.subplots(10, figsize=(3, 20))
fig.tight_layout()

for (data,itr) in zip(dataset, range(len(dataset))):
    raw_dataset = pd.read_csv(str(data)+'_'+'BKM10'+'_'+'pseudodata_generation_from_file.csv', skiprows=1, names=column_names,
                              na_values='?', comment='\t',
                              sep=',', skipinitialspace=True)
    del raw_dataset['count']
    del raw_dataset['Formalism']
    dataset = raw_dataset.copy()

    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('F')
    test_labels = test_features.pop('F')


    # Normalization

    normalizer = tf.keras.layers.Normalization(axis=-1) # Normalization layer
    normalizer.adapt(np.array(train_features)) # Fit the state of the preprocessing later to the data


    # DNN with Multiple Inputs

    def build_and_compile_model(norm):
      model = keras.Sequential([
          norm,
          layers.Dense(64, activation='relu'),
          layers.Dense(64, activation='relu'),
          layers.Dense(1)
      ])

      model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
      return model

    dnn_model = build_and_compile_model(normalizer)

    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=0, epochs=100)
    
    test_results = {}
    test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
    pd.DataFrame(test_results, index=['Mean absolute error [F]']).T

    axs[itr].set_title('smear err ='+str(data))
    axs[itr].plot(history.history['loss'], label='loss')
    axs[itr].plot(history.history['val_loss'], label='val_loss')
    axs[itr].legend(loc="upper right")
    axs[itr].text(0.95, 0.01, 'MAE='+str(test_results['dnn_model']),verticalalignment='bottom', horizontalalignment='right',color='green', fontsize=10)
    axs[itr].set_xlabel('Epoch')
    axs[itr].set_ylabel('Error [F]')



# # Analysis

# test_results = {}
# test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

# pd.DataFrame(test_results, index=['Mean absolute error [F]']).T
# print('Mean absolute error [F] =', test_results['dnn_model'])

# test_predictions = dnn_model.predict(test_features).flatten()

# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, test_predictions)
# plt.xlabel('True Values [F]')
# plt.ylabel('Predictions [F]')
# lims = [0, 3.5]
# plt.xlim(lims)
# plt.ylim(lims)
# _ = plt.plot(lims, lims)

# error = test_predictions - test_labels
# plt.hist(error, bins=20)
# plt.xlabel('Prediction Error [F]')
# plt.ylabel('Count')