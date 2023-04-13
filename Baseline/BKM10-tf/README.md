
ComparisonPlot README
The ComparisonPlot class is designed to create a comparison plot for data located in CSV files in a specified folder path. The plot will show both the fitted values and the true values for a given column in the CSV files.

PARAMETERS

cff: The column name of the variable being plotted.
x_range: The range of x-axis values to plot. default set from 1 to whatever the length of the column is
y_range: The range of y-axis values to plot.
folder_path: The folder path containing the CSV files. 
true_color: The color of the true values points. default set to red
fit_color: The color of the fitted values points and error bars. default set to blue
point_size: The size of the points in the plot. default set to 5
show_error_bars: Boolean indicating whether to show error bars on the fitted values points. default set to True
show_mean_points: Boolean indicating whether to show the fitted values points. default set to True
save_path: The folder path to save the plot in. default set to None
show_plot: Boolean indicating whether to display the plot. default set to True
file_name: The default name of the plot file. If save_path is provided, the plot will be saved using this name with additional information appended. default set to "compare_plot" if save_path is not None
true_value_path: the location of the file that the true values are being read from. if not specified then assumes within same folder.

METHODS

plot(): Plots the comparison plot based on the provided parameters.

EXAMPLE USAGE

python
Copy code

--

SIMPLE PLOT

from ComparisonPlot import ComparisonPlot

# Create ComparisonPlot object
plot = ComparisonPlot(cff='dvcs', folder_path='data/')

# Plot the comparison plot
plot.plot()

In this example above, a comparison plot will be created for the dvcs column, using the fitted values from the csv files in data/ folder.
The plot will use the true values from the default file name "pseudo_KM15" within the current folder.
The plot will default filter the data between x range of 1 to whatever the length of file is that is being read. 
The True value will default to red. The fitted value will defual to blue, and it will show error bars.
The size of the points will be set to 5 as default.
The plot will be displayed on the screen.
The plot will not save the file to any specific folder, nor will it have any filtering for the y range.

--

SPECIFC PLOT

from ComparisonPlot import ComparisonPlot

# Create ComparisonPlot object
plot = ComparisonPlot(cff='ReH', 
                      x_range=[20, 80], 
                      y_range=(-1, 1), 
                      folder_path='data/', 
                      true_color='yellow', 
                      fit_color='green', 
                      point_size=1, 
                      show_error_bars=False, 
                      show_mean_points=True, 
                      save_path='plots/', 
                      show_plot=True, 
                      file_name="comparison_plot"
		      true_value_path='true_data/')

# Plot the comparison plot
plot.plot()

In this example, a comparison plot will be created for the ReH column using the fitted values from the CSV files located in the data/ folder. 
It will use the true values from the specified folder true_data.
The plot will be filtered to include only x-axis values between 1 and 195. 
The y-axis will only show the range -1 to 1 as specified by y_range. 
The true values points will be displayed in yellow and the fitted values points and will be displayed in green. the error bars will not be displayed. 
The size of the points will be 1 and both the fitted values points and error bars will be shown. 
The plot will be saved in the plots/ folder with the default name comparison_plot_{cff}_{num_files}_replicas_range{x_range}.png, 
where num_files is the number of CSV files in the folder. Finally, the plot will be displayed on screen.