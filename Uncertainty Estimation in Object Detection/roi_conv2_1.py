import os as os
import numpy as np
from tqdm import tqdm

for i in tqdm(os.listdir('./detection-results-cow_dataset_val')):
    f = open('./detection-results-cow_dataset_val/'+i, 'r')
    boxes = f.read()
    boxes = boxes.split('\n')[:-1]
    f.close()
    
    os.mkdir('ROI_conv2_1_val/'+i[:-4])
    
    activations = np.load('./activations_conv2_1_val/'+i[:-4]+'.npz')['arr_0']
    mappings = []
    
    counter = 1
    for box in boxes:
        clas, _, xmin, ymin, xmax, ymax = box.split(' ')
        
        new_xmin = float(xmin)*15/640
        new_ymin = float(ymin)*15/480
        new_xmax = float(xmax)*15/640
        new_ymax = float(ymax)*15/480
        
        mapping = activations[int(new_ymin): int(new_ymax), int(new_xmin): int(new_xmax), :]
        
        mappings.append(mapping)
        
        height = mapping.shape[0]
        width = mapping.shape[1]
        channels = mapping.shape[2]
        
        roi = np.zeros((5, 5, channels))
        
        for c in range(channels):
            for row in range(5):
                for col in range(5):
                    try:
                        if row == 4 and col == 4:
                            roi[row][col][c] = np.amax(mapping[row*(height//5): , col*(width//5): , c])
                        elif row == 4 and col != 4:
                            roi[row][col][c] = np.amax(mapping[row*(height//5): , col*(width//5): (col+1)*(width//5), c])
                        elif col==4 and row != 4:
                            roi[row][col][c] = np.amax(mapping[row*(height//5): (row+1)*(height//5), col*(width//5): , c])
                        else:
                            roi[row][col][c] = np.amax(mapping[row*(height//5): (row+1)*(height//5), col*(width//5): (col+1)*(width//5), c])
                    except:
                        pass
                    
                    
        np.save('ROI_conv2_1_val/'+i[:-4]+'/'+str(counter)+'.npy', roi)
        counter+=1