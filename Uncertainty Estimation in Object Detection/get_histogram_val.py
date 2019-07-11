from tqdm import tqdm
import os as os
import cv2
import pandas as pd
import numpy as np

def iou(a, b, epsilon=1e-5):
    
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    width = (x2 - x1)
    height = (y2 - y1)
    
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined + epsilon)
    return iou

def gethist(all_scores):
    if all_scores == []:
        hist = [0 for k in range(10)]
    else:  
        hist,_ = np.histogram(np.array(all_scores),10,[0,1])
    return hist

for image_name in tqdm(os.listdir('./all_boxes_val_coco')[:]):
    f = open('./all_boxes_val_coco/'+image_name, 'r')
    pred = f.read()
    f.close()
    
    pred = pred.split('\n')[:-1]

    output_dict = {}
    output_dict['detection_boxes'] = []
    output_dict['detection_classes'] = []
    output_dict['detection_scores'] = []
    
    for j in range(len(pred)):
        
        l = pred[j].split(' ')
        output_dict['detection_classes'].append(int(l[0]))
        output_dict['detection_scores'].append(float(l[1]))
        output_dict['detection_boxes'].append([float(l[2]), float(l[3]), float(l[4]), float(l[5])])
        
    

    final_boxes_with_variance = {}
    final_boxes_with_variance['detection_boxes'] = []
    final_boxes_with_variance['detection_classes'] = []
    final_boxes_with_variance['detection_scores'] = []
    final_boxes_with_variance['all_scores'] = []
  
    theta = 0.0 # Threshold
    
    marked = [0 for i in range(len(output_dict['detection_boxes']))] #marked array to keep track of boxes considered
  
    while 0 in marked:
        index = marked.index(0)
        marked[index] = 1
        
        selected_box = output_dict['detection_boxes'][index]
        nms_removed_boxes_confidence = []
        class_selected_box = output_dict['detection_classes'][index]
    
        for i in range(len(output_dict['detection_boxes'])):
            #check a box only if it is not already considered and belong to same class as that of selected box
            if marked[i]==0 and output_dict['detection_classes'][i] == class_selected_box:
                to_be_removed_box = output_dict['detection_boxes'][i]
            
                if iou(selected_box, to_be_removed_box) >= 0.600000023842 and output_dict['detection_scores'][i] >= theta: # 0.6 threshold was used by the API
                    marked[i] = 1
                    nms_removed_boxes_confidence.append(output_dict['detection_scores'][i])
  
        nms_removed_boxes_confidence.append(output_dict['detection_scores'][index])
    
   
        final_boxes_with_variance['detection_boxes'].append(selected_box)
        final_boxes_with_variance['detection_classes'].append(class_selected_box)
        final_boxes_with_variance['detection_scores'].append(output_dict['detection_scores'][index])
        final_boxes_with_variance['all_scores'].append(nms_removed_boxes_confidence)

    
    hists = [gethist(all_scores) for all_scores in final_boxes_with_variance['all_scores']]
    #print(hists)

    entry = [[image_name]+list(i) for i in hists]
    
    df = pd.DataFrame(entry, columns = ['image_name', '0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8','0.8-0.9', '0.9-1.0'])
    df.to_csv('./histograms_val/'+image_name[:-4]+'.csv')

 
