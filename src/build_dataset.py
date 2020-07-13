#!/usr/bin/env python
# coding: utf-8


# In[1]:
#################################################################################################################
# Object detection and instance segmentation of planktonic organisms using Mask R-CNN for real-time in-situ analysis
# Author: Sondre Bergum
# email: sondreab@stud.ntnu.no
#
# Date created: 13. Jul 2020
#
# Project: AILARON
# Contact
# email: annette.stahl@ntnu.no
# email: aya.saad@ntnu.no
# funded by RCN IKTPLUSS program (project number 262701) and supported by NTNU AMOS
# Copyright @NTNU 2020
#######################################
#
# THIS FILE AND COE IS STRUCTURED FOR JUPYTER NOTEBOOKS
# DEPENDENT ON:
# https://github.com/facebookresearch/detectron2
# https://github.com/emlynjdavies/PySilCam


import torch, torchvision
print(torch.__version__)

# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os
import numpy as np
import cv2
import json
import csv
import itertools
import random
import collections
from utils import *

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from pysilcam.config import PySilcamSettings
from pysilcam.process import extract_roi

from datetime import datetime


# In[2]:


DIRECTORY = '/home/sondreab/Desktop/DATA/copepod_lab_petridish'
DATA_DIR = DIRECTORY + '/copepods'
#DATA_DIR = False
EXPORT_DIR = DIRECTORY + '/export'
STATS_FILE = 'copepods-STATS.csv'


VISUALIZE_DIR = DIRECTORY + '/visualize/copepods_2020_07_11'
INFERENC_DIR = DIRECTORY + '/inference'

DATASET = 'copepod_stats'


# In[3]:


os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[4]:


def extract_pixels(im, bbox):
    ''' given a binary image (im) and bounding box (bbox[x_min, y_min, x_max, y_max]), this will return all activated pixel coordinates in x and y

    returns:
      all_points_x, all_points_y
    '''
    

    roi = im[ bbox[1]:bbox[3], bbox[0]:bbox[2], 0] # bbox[row, column]
       
    rows = bbox[3] - bbox[1]
    coloumns = bbox[2] - bbox[0]
    
    #print(im.shape)
    #print(roi.shape)
    #print('({}, {})'.format(rows, coloumns))
    #print('iterating')

    
    contours,_= cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = np.array(contours)
    """
    print(type(contours))
    print(contours[0,:,0,:])
    cv2.imshow("visualizer", roi)
    cv2.waitKey(0)
    """
    all_points_x, all_points_y = [], []
    #print(len(contours.shape))
    if len(contours.shape) == 4:
        all_points_x, all_points_y = bbox[0] + contours[0,:,0,0], bbox[1] + contours[0,:,0,1]
    else:
        '''
        print(contours[0].shape)
        cv2.imshow("visualizer", roi)
        cv2.waitKey(0)
        '''
    
        for contour in contours:
            if len(contour[:,0,0]) > len(all_points_x):
                all_points_x = bbox[0] + contour[:,0,0]
                all_points_y = bbox[1] + contour[:,0,1]
        #print(all_points_x)
        
    #print(all_points_x)
    #print(all_points_y)
    
    
    '''
    for r in range(rows):
        #print('(r: {})'.format(r))
        for c in range(coloumns):
            #print('(c: {})'.format(c))
            if roi[r,c] == 255:
                all_points_x.append(bbox[1] + c)
                all_points_y.append(bbox[0] + r)
    '''
    return all_points_x, all_points_y


# In[5]:


def read_stats(directory = DIRECTORY):
    csv_file = os.path.join(directory+'/proc/', STATS_FILE)
    stats = collections.defaultdict(dict)
    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for rows in reader:
            #for key in rows.keys():
            stats[rows['export name'].split('-')[0]][rows['particle index']] = rows
    
    return stats


# In[6]:


def build_annotation_dictionary(directory = DIRECTORY, size_threshold = 0):
    stats = read_stats(directory=directory)
    # parts on the tutorial found at https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
    #config_file = os.path.join(directory,'config.ini')
    #settings = PySilcamSettings(config_file)

    height = 2050
    width = 2448

    train = []
    val = []
    dataset = []
    print('Image:')
    for image, particle in stats.items():
        if image == 'not_exported':
            continue
        print('\t{}'.format(image))
        record = {}
        objects = []
        record["file_name"] = os.path.join(DATA_DIR, image +'.bmp') #.split('/')[-1]
        if not DATA_DIR:
            record["original_file"] = ''
        else:
            record["original_file"] = os.path.join(DATA_DIR, image + '.bmp')
        record["image_id"] = image
        record["height"] = height
        record["width"] = width
        #print('\tParticle:')
        for index, fields in particle.items():
            size = float(fields["equivalent_diameter"])
            if size < size_threshold:
                continue
            
            #print('\t{}'.format(index))
            
            probabilities = np.array([ 
                              float(fields['probability_oil']),
                              float(fields['probability_other']),
                              float(fields['probability_bubble']),
                              float(fields['probability_faecal_pellets']), 
                              float(fields['probability_copepod']),
                              float(fields['probability_diatom_chain']),
                              float(fields['probability_oily_gas'])
                              
                            ])
                
            #print(probabilities)
            class_probability = np.amax(probabilities)
            class_id = np.argmax(probabilities)
            
            probabilities = [ 
                              float(fields['probability_oil']),
                              float(fields['probability_other']),
                              float(fields['probability_bubble']),
                              float(fields['probability_faecal_pellets']), 
                              float(fields['probability_copepod']),
                              float(fields['probability_diatom_chain']),
                              float(fields['probability_oily_gas'])
                            ]
            

            #if the class is lower than the desired threshold of confidence, skip adding the object
            #if (class_probability < 0.7)):                #       settings.Process.threshold)):
            #    continue
            
            minr, minc, maxr, maxc = fields['minr'], fields['minc'], fields['maxr'], fields['maxc']
            xmin = int(float(minc))
            ymin = int(float(minr))
            xmax = int(float(maxc))
            ymax = int(float(maxr))
            
            box_width = xmax - xmin
            box_heigth = ymax - ymin
            bbox = [xmin, ymin, xmax, ymax]
            
            im = cv2.imread(os.path.join(EXPORT_DIR, image+'-SEG.bmp'))
            if im is None:
                print('Image: {} not found, skipping'.format(os.path.join(EXPORT_DIR, image+'-SEG.bmp')))
                continue
            px, py = extract_pixels(im, bbox)
            poly = list(itertools.chain.from_iterable([(x + 0.5, y + 0.5) for x, y in zip(px, py)]))
            obj = {
                "bbox": bbox,
                #"bbox_mode": BoxMode.XYXY_ABS, JSON cannot seriablize structure.BoxMode, this is set to each object when reading json file.
                "segmentation": [poly],
                "category_id": int(class_id),
                "probability" : class_probability,
                "iscrowd": 0
            }
            #print(class_id)
            objects.append(obj)
        record["annotations"] = objects
        dataset.append(record)
    #create_json_file(dataset, DATASET, DIRECTORY)
    return dataset


# In[7]:


def create_json_file(data, file_name, directory=EXPORT_DIR):
    json_file = os.path.join(directory, file_name + '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# In[8]:


def read_json_file(file_name, directory=EXPORT_DIR):
    json_file = os.path.join(directory, file_name+'.json')
    with open(json_file) as f:
        dataset = json.load(f)
    for record in dataset:
        for obj in record['annotations']:
            obj["bbox_mode"] = BoxMode.XYXY_ABS
    return dataset


# In[9]:


def save_dataset_visualization(dataset, directory=VISUALIZE_DIR):
    savepath = VISUALIZE_DIR
    print("Savepath: {}".format(savepath))
    os.makedirs(savepath, exist_ok=True)
    copepod_metadata = MetadataCatalog.get(dataset)
    dataset_dicts = read_json_file(dataset, DIRECTORY)
    print('Saving dataset '+ dataset)
    for image in dataset_dicts:
        img = cv2.imread(image["file_name"])
        if img is None:
                print('Image: {} not found, skipping'.format(image["file_name"]))
                continue
        visualizer = Visualizer(img[:, :, ::-1], metadata=copepod_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(image)
        
        cv2.imwrite(os.path.join(savepath, image['image_id'].split('/')[-1] + '-IMC' + '.png'), vis.get_image()[:, :, ::-1])
        print(image['image_id'].split('/')[-1]+ ' saved!')


# In[13]:


dataset = 'copepod_stats'


# In[ ]:


data = build_annotation_dictionary(size_threshold = 20)

create_json_file(data, dataset, DIRECTORY)
#read_json_file('annotations')


# In[14]:


thing_classes = ['oil', 'other', 'bubble', 'faecal_pellets', 'copepod', 'diatom_chain', 'oily_gas']
DatasetCatalog.register(dataset, lambda d=dataset: read_json_file(d, DIRECTORY))
MetadataCatalog.get(dataset).set(thing_classes=thing_classes)


# In[15]:


save_dataset_visualization(dataset, VISUALIZE_DIR)

