import os as os
import pandas as pd
from tqdm import tqdm

csv_files = os.listdir('./Train_csv')
df = pd.read_csv('./Train_csv/'+csv_files[0], usecols = ['image_name', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0', 'target'])

df_array = []

for i in tqdm(csv_files[1:]):
    df_array.append(pd.read_csv('./Train_csv/'+i, usecols = ['image_name', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0', 'target']))

df = pd.concat([df]+df_array, axis = 0)
df.to_csv('./train_cow_dataset.csv')
