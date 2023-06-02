import pandas as pd
import glob
import os

os.chdir("/project/ptgroup/Devin/Neural_Network/Testing_Data_v5")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "Sample_Data_1M.csv", index=False)