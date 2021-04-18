# Global Fitting Schema
# Nathan Snyder

import json



class GlobalFittingSchema:
    def __init__(self):
        self.data = dict()

    # Read a file into `data`
    def importFromFile(self, filename: str):
        f = open(filename, 'r')
        self.data = json.load(f)
        f.close()

    # Write whatever is currently in `data` to a file
    def writeToFile(self, filename: str):
        f = open(filename, 'w')
        f.write(json.dumps(self.data))
        f.close()

    # Returns a list of the architectures in the data
    def getArchitectures(self):
        return [architecture for architecture in self.data]

    # Returns a list of the optimizers in the data for a given architecture
    def getOptimizers(self, architecture: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")

        return [optimizer for optimizer in self.data[architecture]]

    # Returns a list of the learning rates (as floats) in the data for a given architecture and optimizer
    def getLearningRates(self, architecture: str, optimizer: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not optimizer in self.data[architecture]:
            raise Exception("Optimizer \'" + optimizer + "\' not found in data for architecture \'" + architecture + "\'")

        return [float(learningRate) for learningRate in self.data[architecture][optimizer]]

    # Returns a list of the CFFs in the data for a given architecture, optimizer, and learning rate
    # The learning rate can be given as a float or a string
    def getCFFs(self, architecture: str, optimizer: str, learningRate):
        if (type(learningRate) == int) or (type(learningRate) == float):
            learningRate = str(learningRate)

        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not optimizer in self.data[architecture]:
            raise Exception("Optimizer \'" + optimizer + "\' not found in data for architecture \'" + architecture + "\'")
        if not learningRate in self.data[architecture][optimizer]:
            raise Exception("Learning rate \'" + learningRate + "\' not found in data for optimizer \'" + optimizer + "\'")

        return [cff for cff in self.data[architecture][optimizer][learningRate]]

    # Returns a list of the statistics in the data for a given architecture, optimizer, learning rate, and CFF
    # The learning rate can be given as a float or a string
    def getStatistics(self, architecture: str, optimizer: str, learningRate, cff: str):
        if (type(learningRate) == int) or (type(learningRate) == float):
            learningRate = str(learningRate)

        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not optimizer in self.data[architecture]:
            raise Exception("Optimizer \'" + optimizer + "\' not found in data for architecture \'" + architecture + "\'")
        if not learningRate in self.data[architecture][optimizer]:
            raise Exception("Learning rate \'" + learningRate + "\' not found in data for optimizer \'" + optimizer + "\'")
        if not cff in self.data[architecture][optimizer][learningRate]:
            raise Exception("CFF \'" + cff + "\' not found in data for learning rate \'" + learningRate + "\'")

        return [statistic for statistic in self.data[architecture][optimizer][learningRate][cff]]

    # Returns the value (as a float) of the given architecture, optimizer, learning rate, CFF, and statistic from the data
    # The learning rate can be given as a float or a string
    def getStatistic(self, architecture: str, optimizer: str, learningRate, cff: str, statistic: str):
        if (type(learningRate) == int) or (type(learningRate) == float):
            learningRate = str(learningRate)

        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not optimizer in self.data[architecture]:
            raise Exception("Optimizer \'" + optimizer + "\' not found in data for architecture \'" + architecture + "\'")
        if not learningRate in self.data[architecture][optimizer]:
            raise Exception("Learning rate \'" + learningRate + "\' not found in data for optimizer \'" + optimizer + "\'")
        if not cff in self.data[architecture][optimizer][learningRate]:
            raise Exception("CFF \'" + cff + "\' not found in data for learning rate \'" + learningRate + "\'")
        if not statistic in self.data[architecture][optimizer][learningRate][cff]:
            raise Exception("Statistic \'" + statistic + "\' not found in data for CFF \'" + cff + "\'")

        return float(self.data[architecture][optimizer][learningRate][cff][statistic])

    # Returns a list of statistics (as floats) for the given architecture, optimizer, and cff over all of the learning rates
    # The values in the returned list correspond to the values in getLearningRates(architecture, optimizer)
    def getStatisticForAllLearningRates(self, architecture: str, optimizer: str, cff: str, statistic: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not optimizer in self.data[architecture]:
            raise Exception("Optimizer \'" + optimizer + "\' not found in data for architecture \'" + architecture + "\'")

        ret : list = []
        for learningRate in self.data[architecture][optimizer]:
            if not cff in self.data[architecture][optimizer][learningRate]:
                raise Exception("CFF \'" + cff + "\' not found in data for learning rate \'" + learningRate + "\'")
            if not statistic in self.data[architecture][optimizer][learningRate][cff]:
                raise Exception("Statistic \'" + statistic + "\' not found in data for CFF \'" + cff + "\'")

            ret.append(self.data[architecture][optimizer][learningRate][cff][statistic])
        return ret
    
    # Returns a list of statistics (as floats) for the given architecture, learning rate, and cff over all of the optimizers
    # The values in the returned list correspond to the values in getOptimizers(architecture, learningRate)
    def getStatisticForAllOptimizers(self, architecture: str, learningRate, cff: str, statistic: str):
        if (type(learningRate) == int) or (type(learningRate) == float):
            learningRate = str(learningRate)
        
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")

        ret : list = []
        for optimizer in self.data[architecture]:
            if not learningRate in self.data[architecture][optimizer]:
                raise Exception("Learning rate \'" + learningRate + "\' not found in data for optimizer \'" + optimizer + "\'")
            if not cff in self.data[architecture][optimizer][learningRate]:
                raise Exception("CFF \'" + cff + "\' not found in data for learning rate \'" + learningRate + "\'")
            if not statistic in self.data[architecture][optimizer][learningRate][cff]:
                raise Exception("Statistic \'" + statistic + "\' not found in data for CFF \'" + cff + "\'")

            ret.append(self.data[architecture][optimizer][learningRate][cff][statistic])
        return ret

    # Adds a list of learning rates to the data for a given architecture and optimizer
    # The learning rates can be given as floats or strings
    def addLearningRates(self, architecture: str, optimizer: str, learningRates: list):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not optimizer in self.data[architecture]:
            raise Exception("Optimizer \'" + optimizer + "\' not found in data for architecture \'" + architecture + "\'")

        for learningRate in learningRates:
            if (type(learningRate) == int) or (type(learningRate) == float):
                learningRate = str(learningRate)

            if learningRate in self.data[architecture][optimizer]:
                raise Exception("Learning rate \'" + learningRate + "\' is already in data for optimizer \'" + optimizer + "\'")

            self.data[architecture][optimizer][learningRate] = dict()

            for cff in ['ReH', 'ReE', 'ReHtilde']:
                self.data[architecture][optimizer][learningRate][cff] = dict()

    # Adds a list of optimizers and learning rates to the data for a given architecture
    # The learning rates can be given as floats or strings
    def addOptimizers(self, architecture: str, optimizers: list, learningRates: list):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")

        for optimizer in optimizers:
            if optimizer in self.data[architecture]:
                raise Exception("Optimizer \'" + optimizer + "\' is already in data for architecture \'" + architecture + "\'")

            self.data[architecture][optimizer] = dict()

            if len(learningRates) > 0:
                self.addLearningRates(architecture, optimizer, learningRates)

    # Adds an architecture and a list of optimizers and learning rates to the data
    # The learning rates can be given as floats or strings
    def addArchitecture(self, architecture: str, optimizers: list, learningRates: list):
        if architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' is already in the data")

        self.data[architecture] = dict()

        if len(optimizers) > 0:
            self.addOptimizers(architecture, optimizers, learningRates)

    # Set the value of a specified statistic in the data
    def setStatistic(self, architecture: str, optimizer: str, learningRate, cff: str, statistic: str, value):
        if (type(learningRate) == int) or (type(learningRate) == float):
            learningRate = str(learningRate)

        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not optimizer in self.data[architecture]:
            raise Exception("Optimizer \'" + optimizer + "\' not found in data for architecture \'" + architecture + "\'")
        if not learningRate in self.data[architecture][optimizer]:
            raise Exception("Learning rate \'" + learningRate + "\' not found in data for optimizer \'" + optimizer + "\'")
        if not cff in self.data[architecture][optimizer][learningRate]:
            raise Exception("CFF \'" + cff + "\' not found in data for learning rate \'" + learningRate + "\'")

        self.data[architecture][optimizer][learningRate][cff][statistic] = value
