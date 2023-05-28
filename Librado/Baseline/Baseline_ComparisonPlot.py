#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class ComparisonPlot:
    def __init__(self, cff, true_color='r', fit_color='b', point_size=5, show_error_bars=True, show_mean_points=True, save_path=None, show_plot=True, y_range=None, x_range=None,  folder_path=None,  file_name="comparison_plot", true_value_path=None):
        self.cff = cff
        self.x_range = x_range
        self.y_range = y_range
        self.folder_path = folder_path
        self.fit_color = fit_color
        self.true_color = true_color
        self.point_size = point_size
        self.show_error_bars = show_error_bars
        self.show_mean_points = show_mean_points
        self.save_path = save_path
        self.show_plot = show_plot
        self.file_name = file_name
        self.true_value_path = true_value_path if true_value_path is not None else "pseudo_KM15.csv"
        
    def plot(self):
        if self.folder_path is not None:
            # Find all the files in the folder_path that start with "bySetCFFs"
            files = [f for f in os.listdir(self.folder_path) if f.startswith("bySetCFFs")]
            # Read the CSV files into a list of DataFrames
            dfs = [pd.read_csv(os.path.join(self.folder_path, f), index_col=0) for f in files]
            
            # Count the number of files read through the folder
            num_files = len(dfs)
            
            # Concatenate the DataFrames along the third dimension
            array_3d = np.stack([df.values for df in dfs], axis=-1)
            # Compute the mean and standard deviation along the third dimension
            array_mean = np.mean(array_3d, axis=-1)
            array_stdev = np.std(array_3d, axis=-1)
            
            # Convert the resulting NumPy arrays into DataFrames and set the column and index names
            df_mean = pd.DataFrame(array_mean, columns=dfs[0].columns, index=dfs[0].index)
            df_stdev = pd.DataFrame(array_stdev, columns=dfs[0].columns, index=dfs[0].index)
            
            # Shift the index of both DataFrames by 1
            df_mean.index = df_mean.index + 1
            df_stdev.index = df_stdev.index + 1
            
            # Convert the mean and stdev arrays to DataFrames and set the column and index names
            df_mean.columns = ['ReH', 'ReE', 'ReHTilde', 'dvcs']
            df_stdev.columns = ['ReH', 'ReE', 'ReHTilde', 'dvcs']
            
            #plot x range default
            if self.x_range is None:
                self.x_range = [1,np.shape(df_mean)[0]]
        else:
            result = pd.read_csv(self.true_value_path)
            self.x_range = [1,result.iloc[-1,0]]

        # Filter the data based on x range
        result = pd.read_csv(self.true_value_path)
        result_filtered = pd.read_csv(self.true_value_path)[(result['#Set'] >= self.x_range[0]) & (result['#Set'] <= self.x_range[1])]
        if self.folder_path is not None:
            df_filtered = df_mean[(df_mean.index >= self.x_range[0]) & (df_mean.index <= self.x_range[1])].reset_index()
            y_err_filtered = df_stdev[(df_stdev.index >= self.x_range[0]) & (df_stdev.index <= self.x_range[1])][self.cff].to_numpy()

        # Plot filtered mean and filtered errorbars
        if self.folder_path is not None and self.show_mean_points:
            plt.scatter(np.array(df_filtered.iloc[:, 0]), np.array(df_filtered[self.cff]), color=self.fit_color, s=self.point_size)
        if self.folder_path is not None and self.show_error_bars:
            plt.errorbar(np.array(df_filtered.iloc[:, 0]), np.array(df_filtered[self.cff]), color=self.fit_color, yerr=y_err_filtered, fmt='none', capsize=3, label = 'Fitted Values')

        # Plot x against y
        plt.scatter(result_filtered['#Set'], result_filtered[self.cff], color=self.true_color, s=self.point_size, label='True Values')

        # Modify the plot title based on the number of files and column_y
        if self.folder_path is not None:
            plt.title(f'{self.cff} Comparison Plot for {num_files} Replicas (range:{self.x_range})')
        else:
            plt.title(f'{self.cff} Plot (range:{self.x_range})')
        plt.legend()
        plt.xlabel('Bin')
        plt.ylabel(f'{self.cff}')

        # Filter the plot based on y range
        if self.y_range is not None:
            plt.ylim(self.y_range)

        # Saves plot to path and sets default name of file
        if self.save_path is not None:
            self.file_name = f"{self.file_name}_{self.cff}"
            if self.folder_path is not None:
                self.file_name += f"_{num_files}_replicas_range_{self.x_range}"
            else:
                self.file_name += f"_replicas_range_{self.x_range}"
            plt.savefig(os.path.join(self.save_path, self.file_name))

        # Show plot
        if self.show_plot:
            plt.show()

