import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.spatial.distance import euclidean
import os

help_epilogue = """
Example usage:
--------------
$ python your_script.py cff_name --folder_path /path/to/folder --y_range 0,10 --x_range 1,100 --save_path /path/to/save --hide_mean_points --hide_plot

This script plots a comparison graph for a specific CFF (Central Form Factor) using the provided options.
The available options are:

cff_name            Name of the CFF to plot
--folder_path       Path to the folder containing the CSV files (optional)
--y_range           Range for the y-axis (format: min,max) (optional)
--x_range           Range for the x-axis (format: min,max) (optional)
--save_path         Path to save the plot image (optional)
--hide_mean_points  Do not show mean points (optional)
--hide_plot         Do not display the plot (optional)
"""


class ComparisonPlot:
    def __init__(self, cff, true_color='r', fit_color='b', point_size=5, show_error_bars=True, show_mean_points=True, save_path=None, show_plot=True, y_range=None, x_range=None, folder_path=None, file_name="comparison_plot", true_value_path=None):
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
        self.true_value_path = true_value_path if true_value_path is not None else "pseudoKM15_New_FormFactor.csv"


    def calculate_similarity(self, signal1, signal2):
        similarity = 1 / (1 + euclidean(signal1, signal2))
        return similarity

    def calculate_rmse(self, true_values, predicted_values):
        squared_diff = (true_values - predicted_values) ** 2
        mse = np.mean(squared_diff)
        rmse = np.sqrt(mse)
        return rmse

    def plot(self):
        if self.folder_path is not None:
            # Find all the files in the folder_path that start with "bySetCFFs"
            files = [f for f in os.listdir(self.folder_path) if f.startswith("bySetCFFs")]
            # Read the CSV files into a list of DataFrames
            dfs = [pd.read_csv(os.path.join(self.folder_path, f), index_col=0, nrows=195) for f in files]

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

            # Plot x range default
            if self.x_range is None:
                self.x_range = [1, np.shape(df_mean)[0]]
        else:
            result = pd.read_csv(self.true_value_path)
            self.x_range = [1, result.iloc[-1, 0]]

        # Filter the data based on x range
        result = pd.read_csv(self.true_value_path)
        result = result.groupby('#Set').first().reset_index()
        result_filtered = result[
            (result['#Set'] >= self.x_range[0]) & (result['#Set'] <= self.x_range[1])]

        if self.folder_path is not None:
            df_filtered = df_mean[(df_mean.index >= self.x_range[0]) & (df_mean.index <= self.x_range[1])].reset_index()
            y_err_filtered = df_stdev[(df_stdev.index >= self.x_range[0]) & (df_stdev.index <= self.x_range[1])][self.cff].to_numpy()

        # Plot filtered mean and filtered errorbars
        if self.folder_path is not None and self.show_mean_points:
            plt.scatter(np.array(df_filtered.iloc[:, 0]), np.array(df_filtered[self.cff]), color=self.fit_color,
                        s=self.point_size)
        if self.folder_path is not None and self.show_error_bars:
            plt.errorbar(np.array(df_filtered.iloc[:, 0]), np.array(df_filtered[self.cff]), color=self.fit_color,
                         yerr=y_err_filtered, fmt='none', capsize=3, label='Fitted Values')

        # Plot x against y
        plt.scatter(result_filtered['#Set'], result_filtered[self.cff], color=self.true_color, s=self.point_size,
                    label='True Values')

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

        #Calculate similarity and print to command line
        if self.folder_path is not None:
            similarity = self.calculate_similarity(df_filtered[self.cff], result_filtered[self.cff])
            print(f"Similarity between fitted values and true values: {similarity}")

            rmse = self.calculate_rmse(df_filtered[self.cff], result_filtered[self.cff])
            print(f"RMSE: {rmse}")

            diff = np.abs(df_filtered[self.cff] - result_filtered[self.cff])
            within_std = np.sum(diff <= y_err_filtered)
            precision = (within_std / len(diff)) * 100
            print(f"Precison: {precision}%")
    
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cff", help="Name of the CFF to plot")
    parser.add_argument("--true_color", default="r", help="Color for true values (default: red)")
    parser.add_argument("--fit_color", default="b", help="Color for fitted values (default: blue)")
    parser.add_argument("--point_size", type=int, default=5, help="Size of the data points (default: 5)")
    parser.add_argument("--hide_error_bars", action="store_true", help="Do not show error bars")
    parser.add_argument("--hide_mean_points", action="store_true", help="Do not show mean points")
    parser.add_argument("--save_path", help="Path to save the plot image")
    parser.add_argument("--hide_plot", action="store_true", help="Do not display the plot")
    parser.add_argument("--y_range", help="Range for the y-axis (format: min,max)")
    parser.add_argument("--x_range", help="Range for the x-axis (format: min,max)")
    parser.add_argument("--folder_path", help="Path to the folder containing the CSV files")
    parser.add_argument("--file_name", default="comparison_plot", help="Name of the output file (default: comparison_plot)")
    parser.add_argument("--true_value_path", help="Path to the CSV file containing true values")

    args = parser.parse_args()

    plotter = ComparisonPlot(
        args.cff,
        true_color=args.true_color,
        fit_color=args.fit_color,
        point_size=args.point_size,
        show_error_bars=not args.hide_error_bars,
        show_mean_points=not args.hide_mean_points,
        save_path=args.save_path,
        show_plot=not args.hide_plot,
        y_range=[float(x) for x in args.y_range.split(",")] if args.y_range else None,
        x_range=[float(x) for x in args.x_range.split(",")] if args.x_range else None,
        folder_path=args.folder_path,
        file_name=args.file_name,
        true_value_path=args.true_value_path
    )

    plotter.plot()
