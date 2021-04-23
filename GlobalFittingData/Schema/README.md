# How to Use the Global Fitting Schema

Note: See [schema_test.ipynb](https://github.com/extraction-tools/ANN/blob/master/GlobalFittingData/Schema/schema_test.ipynb) for an example of how to use the global fitting schema

---

&nbsp;

## Instructions for writing data to a file using the global fitting schema

First, import GlobalFittingSchema and create an instance of the GlobalFittingSchema class:
```Python
from GlobalFittingSchema import GlobalFittingSchema
g = GlobalFittingSchema()
```

If the architecture that you're using is not already in the data, add it:
```Python
# Adds the architecture 'test' to the data
g.addArchitecture('test')
```

Next add the run that you are using:
```Python
# Adds the run 'Run1-test-Nathan' to the architecture 'test'
g.addRun('test', 'Run1-test-Nathan')
```

Once you have the run in place, set the hyperparameters that characterize the run:
```Python
g.setHyperparameter('test', 'Run1-test-Nathan', 'optimizer', 'SGD')
g.setHyperparameter('test', 'Run1-test-Nathan', 'learning-rate', 0.3)
```
Any hyperparameter name can be used.

Then run your model and get the data statistics, and add them like so:
```Python
# Sets the Mean Percent Error of ReH for the architecture 'test' and run 'Run1-test-Nathan' to 5
g.setStatistic('test', 'Run1-test-Nathan', 'ReH', 'MPE', 5)

# Sets the Root Mean Square Error of ReE for the architecture 'test' and run 'Run1-test-Nathan' to 9.4
g.setStatistic('test', 'Run1-test-Nathan', 'ReH', 'RMSE', 9.4)
```
Any statistic name can be used.

After all of the statistics have been added, export the data:
```Python
g.writeToFile('output.json')
```

---

&nbsp;

## Instructions for reading data from a file using the global fitting schema

First, import GlobalFittingSchema and create an instance of the GlobalFittingSchema class:
```Python
from GlobalFittingSchema import GlobalFittingSchema
g = GlobalFittingSchema()
```

Import the JSON file containing the data:
```Python
g.importFromFile('input.json')
```

There are functions for getting a list of architectures, runs, etc. from the data:
```Python
g.getArchitectures() # Get a list of architectures in the data
g.getRuns('test') # Get a list of runs in the given architecture
g.getHyperparameters('test', 'Run1-test-Nathan') # Get a list of hyperparameters in the given run
g.getStatistics('test', 'Run1-test-Nathan', 'ReH') # Get a list of statistics for the given CFF, run, and architecture
```

If you want the value of a given hyperparameter, use the `getHyperparameter` function:
```Python
# Get the optimizer value for the given run and architecture
g.getHyperparameter('test', 'Run1-test-Nathan', 'optimizer')
```

If you want the value of a given statistic, use the `getStatistic` function:
```Python
# Get the MPE value for the given CFF, run, and architecture
g.getStatistic('test', 'Run1-test-Nathan', 'ReH', 'MPE')
```
