import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('After_v3.csv')

Err = df.iloc[:,-1:]
P = df.iloc[:,-3:]

plt.figure()
plt.plot(P,Err, '.')
plt.xlabel('Polarization')
plt.ylabel('Relative Percent Error (%)')
plt.savefig('Accuracy_Plot.pdf')