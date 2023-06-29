import pandas as pd
import numpy as np
import os
import glob

os.chdir("/project/ptgroup/Devin/ANN/BKM_T/CFF_Data")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "CFFs.csv", index=False)