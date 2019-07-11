import os as os
import pandas as pd
from tqdm import tqdm

csv_files = os.listdir('./Histograms_val')
df = pd.read_csv('./Histograms_val/'+csv_files[0], usecols = ['image_name', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0'])

df_array = []

for i in tqdm(csv_files[1:]):
    df_array.append(pd.read_csv('./Histograms_val/'+i, usecols = ['image_name', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0']))

df = pd.concat([df]+df_array, axis = 0)
df.to_csv('./val_coco.csv', index = False)
