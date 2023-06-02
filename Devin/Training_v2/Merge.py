import pandas as pd
import glob
import os

os.chdir("/project/ptgroup/Devin/Neural_Network/Sample_Data/Sample_Data_Prediction")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "Sample_Data_500K.csv", index=False)