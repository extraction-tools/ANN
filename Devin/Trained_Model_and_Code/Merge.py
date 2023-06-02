import os
import pandas as pd
import numpy as np

def get_df(path):
    df=pd.DataFrame()
    os.chdir(str(path))
    for file in os.listdir():
        if file.endswith('.csv'):
            aux=pd.read_csv(file)
            df=df.append(aux)
    return df


df=get_df("/project/ptgroup/Devin/Neural_Network/Sample_Data/Sample_Data_1M")
# df_noise = get_df("/project/ptgroup/Devin/Neural_Network/Sample_Data/Sample_Data_5M_Noisy")

df.to_csv(f"Sample_Data_1M.csv",index=False)
# df_noise.to_csv(f"Sample_Data_5M_noisy.csv",index=False)