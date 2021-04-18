**Instructions for writing data to a file using the global fitting schema**

First, import GlobalFittingSchema and create an instance of the GlobalFittingSchema class:
```Python
from GlobalFittingSchema import GlobalFittingSchema
g = GlobalFittingSchema()
```

Add which architectures, optimizers, and learning rates you are using:
```Python
# Adds the architecture 'test' with the optimizers 'Adam' and 'Ftrl' and the learning rates '0.00001' and '0.8123'
g.addArchitecture('test', ['Adam', 'Ftrl'], [0.00001, 0.8123])
```

Once you run your model and get the data statistics, add them like so:
```Python
# Sets the Mean Percent Error of ReH for the architecture 'test', optimizer 'Adam', and learning rate 0.00001 to 5
g.setStatistic('test', 'Adam', 0.00001, 'ReH', 'MPE', 5)

# Sets the Root Mean Square Error of ReE for the architecture 'test', optimizer 'Adam', and learning rate 0.8123 to 9.4
g.setStatistic('test', 'Adam', 0.8123, 'ReE', 'RMSE', 9.4)
```
Any statistic name can be used.

After all of the statistics have been added, export the data:
```Python
g.writeToFile('output.json')
```



**Instructions for reading data from a file using the global fitting schema**

First, import GlobalFittingSchema and create an instance of the GlobalFittingSchema class:
```Python
from GlobalFittingSchema import GlobalFittingSchema
g = GlobalFittingSchema()
```

Import the JSON file containing the data:
```Python
g.importFromFile('input.json')
```

There are functions for getting a list of architectures, optimizers, learning rates, etc. from the data:
```Python
g.getArchitectures() # Get a list of architectures in the data
g.getOptimizers('test') # Get a list of optimizers in the given architectures
g.getLearningRates('test', 'Ftrl') # Get a list of learning rates in the given optimizer and architecture
g.getCFFs('test', 'Ftrl', 0.00001) # Get a list of CFFs in the given leraning rate, optimizer, and architecture
g.getStatistics('test', 'Ftrl', 0.00001, 'ReH') # Get a list of statistics in the given CFF, leraning rate, optimizer, and architecture
```

If you want the value of a given statistic, use the `getStatistic` function:
```Python
g.getStatistic('test', 'Adam', 0.00001, 'ReH', 'MPE')
```

There are also functions for getting a list of statistics across optimizers and learnings rates:
```Python
# Returns a list of MPE statistics for ReE with optimizer SGD across all available learning rates
g.getStatisticForAllLearningRates('test', 'SGD', 'ReE', 'MPE')

# Returns a list of MPE statistics for ReE with learning rate 0.5 across all available optimizers
g.getStatisticForAllOptimizers('test', 0.5, 'ReE', 'MPE')
```

This makes it easy to plot the data:
```Python
import matplotlib.pyplot as plt

learning_rates = g.getLearningRates('test', 'SGD')
MPE_by_learning_rate = g.getStatisticForAllLearningRates('test', 'SGD', 'ReE', 'MPE')

plt.plot(learning_rates, MPE_by_learning_rate)
plt.ylabel('Mean Percent Error')
plt.xlabel('learning rate')
plt.show()
```
