from tqdm import tqdm

import os as os



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



for image_name in tqdm(os.listdir('./all_boxes_train_coco')):

    f = open('./all_boxes_train_coco/'+image_name, 'r')

    pred = f.read()

    f.close()

    

    pred = pred.split('\n')[:9000]



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

  

         #print(nms_removed_boxes_confidence)

    

   

        final_boxes_with_variance['detection_boxes'].append(selected_box)

        final_boxes_with_variance['detection_classes'].append(class_selected_box)

        final_boxes_with_variance['detection_scores'].append(output_dict['detection_scores'][index])

        final_boxes_with_variance['all_scores'].append(nms_removed_boxes_confidence)

    

    f = open('./detection-results-coco-train/'+image_name[:-4]+'.txt', 'w')

    

    for j in range(len(final_boxes_with_variance['detection_boxes'])):

        f.write(str(final_boxes_with_variance['detection_classes'][j])+' ')

        f.write(str(final_boxes_with_variance['detection_scores'][j])+' ')

        f.write(str(final_boxes_with_variance['detection_boxes'][j][0])+' ')

        f.write(str(final_boxes_with_variance['detection_boxes'][j][1])+' ')

        f.write(str(final_boxes_with_variance['detection_boxes'][j][2])+' ')

        f.write(str(final_boxes_with_variance['detection_boxes'][j][3])+'\n')

        

    f.close()
