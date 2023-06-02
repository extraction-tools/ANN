import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import matplotlib.mlab as mlab
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
df = pd.read_csv('0p_v51.csv')

df['P_diff'] = df['P'] - df['P_True']
x = np.array(df['P_diff'])
# bins=df['P_diff'].quantile([0,.05,0.1,0.15,0.20,0.25,0.3,0.35,0.40,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1]).to_list()
# plt.hist(df['P_diff'],bins = bins,range = (-2,2))
# plt.savefig('Histogram.png')


# Implementation of matplotlib function
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
   
# np.random.seed(10**7)
# mu = 121 
# sigma = 21
# x = mu + sigma * np.random.randn(1000)
   
num_bins = 100
   
n, bins, patches = plt.hist(x, num_bins, 
                            density = True, 
                            color ='green',
                            alpha = 0.7)
   
(mu, sigma) = norm.fit(x)

y = norm.pdf(bins, mu, sigma)
l = plt.plot(bins, y, 'r--', linewidth=2)
  
# plt.plot(bins, y, '--', color ='black')
  
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title("Histogram of 0%% Noise: mu=%.3f, sigma=%.3f" %(mu, sigma))
# plt.title(r'$\mathrm{Histogram\ of\ I0%% Noise:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))

plt.grid(True)
  
plt.savefig('Histogram_0p_v51.png')

