import pandas as pd
import os as os
from tqdm import tqdm

for i in tqdm(os.listdir('./Histograms')[:]):
    df1 = pd.read_csv('./Histograms/'+i, usecols = ['image_name', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0'])
    df2 = pd.read_csv('./Targets/'+i, usecols = ['target'])
    
    df3 = pd.concat([df1, df2], axis=1)
    
    df3.to_csv('./Train_csv/'+i+'.csv', index = False)
