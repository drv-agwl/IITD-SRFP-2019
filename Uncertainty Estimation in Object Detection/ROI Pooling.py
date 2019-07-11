import os as os
import numpy as np
import pandas as pd
from tqdm import tqdm

for i in tqdm(os.listdir('./detection-results-cow_dataset_val')[:]):
    f = open('./detection-results-cow_dataset_val/'+i, 'r')
    boxes = f.read()
    boxes = boxes.split('\n')[:-1]
    f.close()
    
    os.mkdir('ROI_val/'+i[:-4])
    
    activations = np.load('./activations_val/'+i[:-4]+'.npy')
    mappings = []
    
    counter = 1
    for box in boxes:
        clas, _, xmin, ymin, xmax, ymax = box.split(' ')
        
        new_xmin = float(xmin)*19/640
        new_ymin = float(ymin)*19/480
        new_xmax = float(xmax)*19/640
        new_ymax = float(ymax)*19/480
        
        mapping = activations[int(new_ymin):int(new_ymax), int(new_xmin):int(new_xmax),:]
        
        mappings.append(mapping)
        
        #ROI
        
        height = mapping.shape[0]
        width = mapping.shape[1]
        channels = mapping.shape[2]
        
        roi = np.zeros((2, 2, channels))
        
        for c in range(channels):
            try:
                roi[0][0][c] = np.amax(mapping[0:height//2, 0:width//2, c])
            except:
                pass
            try:
                roi[0][1][c] = np.amax(mapping[0:height//2, width//2:width, c])
            except:
                pass
            try:
                roi[1][0][c] = np.amax(mapping[height//2:height, 0:width//2, c])
            except:
                pass
            try:     
                roi[1][1][c] = np.amax(mapping[height//2:height, width//2:width, c])
            except:
                pass
            
        np.save('ROI_val/'+i[:-4]+'/'+str(counter)+'.npy', roi)
        counter+=1