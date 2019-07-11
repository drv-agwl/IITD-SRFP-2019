import pandas as pd
from tqdm import tqdm

df1 = pd.read_csv('./val_coco.csv', usecols = ['image_name', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0'])
df2 = pd.read_csv('./predictions_2.csv', usecols = ['predictions'])

print("csv files read")

df3 = pd.concat([df1, df2], axis = 1)

print("concatenated")

df3.to_csv('./histograms_and_predictions_val_2.csv', index = False)
