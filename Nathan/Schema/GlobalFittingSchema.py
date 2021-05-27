# Global Fitting Schema
# Nathan Snyder

import json



class GlobalFittingSchema:
    cffDict : dict = {'ReH': 0, 'ReE': 1, 'ReHtilde': 2}

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

    # Returns a list of runs for a given architecture in the data
    def getRuns(self, architecture: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")

        return [run for run in self.data[architecture]]

    # Returns a list of the hyperparameters for a given architecture and run
    def getHyperparameters(self, architecture: str, run: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not run in self.data[architecture]:
            raise Exception("Run \'" + run + "\' not found for the architecture \'" + architecture + "\'")

        return [hyperparameter for hyperparameter in self.data[architecture][run][0]]

    # Returns the value of a given hyperparameter in a given architecture and run
    def getHyperparameter(self, architecture: str, run: str, hyperparameter: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not run in self.data[architecture]:
            raise Exception("Run \'" + run + "\' not found for the architecture \'" + architecture + "\'")
        if not hyperparameter in self.data[architecture][run][0]:
            raise Exception("Hyperparameter \'" + hyperparameter + "\' not found for the run \'" + run + "\'")

        return self.data[architecture][run][0][hyperparameter]

    # Returns a list of statistics for a given CFF in a given architecture and run
    def getStatistics(self, architecture: str, run: str, cff: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not run in self.data[architecture]:
            raise Exception("Run \'" + run + "\' not found for the architecture \'" + architecture + "\'")

        if cff in GlobalFittingSchema.cffDict:
            return [stat for stat in self.data[architecture][run][1][GlobalFittingSchema.cffDict[cff]]]
        else:
            raise Exception("invalid Compton Form Factor")

    # Returns the value of a given statistic for a given CFF, architecture, and run
    def getStatistic(self, architecture: str, run: str, cff: str, statistic: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not run in self.data[architecture]:
            raise Exception("Run \'" + run + "\' not found for the architecture \'" + architecture + "\'")

        statistics = dict()

        if cff in GlobalFittingSchema.cffDict:
            statistics = self.data[architecture][run][1][GlobalFittingSchema.cffDict[cff]]
        else:
            raise Exception("invalid Compton Form Factor")

        if not statistic in statistics:
            raise Exception("Statistic \'" + statistic + "\' not found for the CFF \'" + cff + "\'")

        return statistics[statistic]

    # Adds a new architecture to the data as long as it's not already in the data
    def addArchitecture(self, architecture: str):
        if not architecture in self.data:
            self.data[architecture] = dict()

    # Adds a new run to the data as long as it's not already in the data for a given architecture
    def addRun(self, architecture: str, run: str):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")

        if not run in self.data[architecture]:
            self.data[architecture][run] = [dict(), [dict(), dict(), dict()]]

    # Sets the value of a hyperparameter for a given run and architecture
    def setHyperparameter(self, architecture: str, run: str, hyperparameter: str, value):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not run in self.data[architecture]:
            raise Exception("Run \'" + run + "\' not found for the architecture \'" + architecture + "\'")

        if not hyperparameter in self.data[architecture][run][0]:
            self.data[architecture][run][0][hyperparameter] = value

    # Set the value of a statistic for a given run and architecture
    def setStatistic(self, architecture: str, run: str, cff: str, statistic: str, value):
        if not architecture in self.data:
            raise Exception("Architecture \'" + architecture + "\' not found in data")
        if not run in self.data[architecture]:
            raise Exception("Run \'" + run + "\' not found for the architecture \'" + architecture + "\'")

        if cff in GlobalFittingSchema.cffDict:
            self.data[architecture][run][1][GlobalFittingSchema.cffDict[cff]][statistic] = value
        else:
            raise Exception("invalid Compton Form Factor")
