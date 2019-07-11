import pandas as pd
import numpy as np
from tqdm import tqdm
import os as os

df = pd.read_csv('./histograms_and_predictions_val_2.csv', usecols = ['image_name', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0', 'predictions'])

for i in tqdm(os.listdir('detection-results-cow_dataset_val')[:]):
    f = open('./detection-results-cow_dataset_val/'+i, 'r')
    s = f.read()
    l = s.split('\n')
 
    image_prediction = np.array(df[df['image_name'] == int(i[:-4])].predictions)

    f2 = open('./detection-results-cow_dataset_val_neural_net_conv2_1/'+i, 'w')
    
    for j in range(len(l)-1):
        l2 = l[j].split(' ')
        f2.write(l2[0]+' ')
        f2.write(str(image_prediction[j])+' ')
        f2.write(l2[2]+' ')
        f2.write(l2[3]+' ')
        f2.write(l2[4]+' ')
        f2.write(l2[5]+'\n')

    f2.close()
    f.close()
