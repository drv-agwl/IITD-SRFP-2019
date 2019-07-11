import os as os
from tqdm import tqdm
from shutil import copy

xmls = []
for i in os.listdir('./total_images'):
    if i.endswith('.xml'):
        xmls.append(i)
        
import random
train_xmls = random.sample(xmls, 2842)

test_xmls = []
for i in os.listdir('./total_images'):
    if i.endswith('.xml') and i not in train_xmls:
        test_xmls.append(i)
        
for i in train_xmls:
    copy('./total_images/'+i, './train/'+i)
    copy('./total_images/'+i[:-4]+'.jpg', './train/'+i[:-4]+'.jpg')


for i in test_xmls:
    copy('./total_images/'+i, './test/'+i)
    copy('./total_images/'+i[:-4]+'.jpg', './test/'+i[:-4]+'.jpg')
