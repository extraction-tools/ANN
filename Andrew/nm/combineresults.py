from os import listdir
from os.path import isfile, join
import pandas as pd

path = "ResultsConfig"+"/"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

configs = pd.read_csv('configs.csv')
totData = pd.DataFrame()
for x in configs['batch'].unique():
    configSet = pd.DataFrame()
    minIndex = configs[configs['batch']==x]['line'].min()
    
    for b in configs[configs['batch']==x]['line']:
        df = pd.read_csv(path+"Results"+str(b))
        df['replica'] = df['replica'] + (b-minIndex)*15
        configSet = configSet.append(df)
    totData = totData.append(configSet)

totData.to_csv("Results.csv")