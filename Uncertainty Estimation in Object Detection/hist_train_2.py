from tqdm import tqdm
import os as os
import numpy as np
import pandas as pd

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


def getTargets(all_scores, detection_box, clas, l):
    
    #coordinates are according to 300x300, scaling them into 640x480
    detection_box[0] = detection_box[0]*(640./300.)
    detection_box[1] = detection_box[1]*(480./300.)
    detection_box[2] = detection_box[2]*(640./300.)
    detection_box[3] = detection_box[3]*(480./300.)
        
    target_value = 0
    
    for j in range(len(l)-1):
      s2 = l[j]
      l2 = s2.split(' ')

      ground_truth_label = l2[0]

      ground_truth_coordinates = [float(x) for x in l2[1:]]

      if iou(detection_box, ground_truth_coordinates) >= 0.5 and ground_truth_label == clas:
        target_value = 1
        break

      elif ground_truth_label == clas and iou(detection_box, ground_truth_coordinates) < 0.5:
        target_value = 0

      elif ground_truth_label != clas:
        target_value = 0

    return target_value

for image_name in tqdm(os.listdir('./all_boxes_cow_dataset')[:]):
    f = open('./all_boxes_cow_dataset/'+image_name, 'r')
    pred = f.read()
    f.close()
    
    pred = pred.split('\n')[:-1]

    output_dict = {}
    output_dict['detection_boxes'] = []
    output_dict['detection_classes'] = []
    output_dict['detection_scores'] = []
    
    for j in range(len(pred)):
        
        l = pred[j].split(' ')
        if l[0] == '20.0':
            output_dict['detection_classes'].append('cow')
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
            
                if iou(selected_box, to_be_removed_box) >= 0.45 and output_dict['detection_scores'][i] >= theta: # 0.45 threshold was used by the API
                    marked[i] = 1
                    nms_removed_boxes_confidence.append(output_dict['detection_scores'][i])
  
        nms_removed_boxes_confidence.append(output_dict['detection_scores'][index])
    
   
        final_boxes_with_variance['detection_boxes'].append(selected_box)
        final_boxes_with_variance['detection_classes'].append(class_selected_box)
        final_boxes_with_variance['detection_scores'].append(output_dict['detection_scores'][index])
        final_boxes_with_variance['all_scores'].append(nms_removed_boxes_confidence)

    f = open('./ground-truth-cow_dataset-train/' + image_name[:-4]+'.txt', 'r')
    s = f.read()
    l = s.split('\n')

    
    targets = [getTargets(all_scores, detection_box, clas, l) for all_scores, detection_box, clas in zip(final_boxes_with_variance['all_scores'], final_boxes_with_variance['detection_boxes'], final_boxes_with_variance['detection_classes'])]

    df = pd.DataFrame(columns = ['image_name', 'target'])
    
    for i in range(len(targets)):
        df.loc[i] = [image_name[:-4]]+[targets[i]]
        
    df.to_csv('./Targets/'+image_name[:-4]+'.csv', index = False)
    
    # for i in range(len(final_boxes_with_variance['detection_boxes'])):
    
    #     if np.array(final_boxes_with_variance['all_scores'][i]) == []:
    #         hist = [[0] for k in range(10)]
    #     else:  
    #         hist = cv2.calcHist([np.array(final_boxes_with_variance['all_scores'][i])],[0],None,[10],[0,1])
        

    #     target_value = 0
        
    #     for j in range(len(l)-1):
    #       s2 = l[j]
    #       l2 = s2.split(' ')

    #       ground_truth_label = l2[0]

    #       ground_truth_coordinates = [float(x) for x in l2[1:]]

    #       if iou(final_boxes_with_variance['detection_boxes'][i], ground_truth_coordinates) >= 0.5 and ground_truth_label == final_boxes_with_variance['detection_classes'][i]:
    #         target_value = 1
    #         break

    #       elif ground_truth_label == final_boxes_with_variance['detection_classes'][i] and iou(final_boxes_with_variance['detection_boxes'][i], ground_truth_coordinates) < 0.5:
    #         target_value = 0

    #       elif ground_truth_label != final_boxes_with_variance['detection_classes'][i]:
    #         target_value = 0


      # df.loc[counter] = [image_name] + [x[0] for x in hist] + [target_value]
      # counter += 1

 