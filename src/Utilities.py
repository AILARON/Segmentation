#!/usr/bin/env python
# coding: utf-8

# In[ ]:
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

import os
import numpy as np
import cv2
import json
import csv
import itertools
import random
import collections


# In[ ]:


"""
## Cutting out pieces of images

inferred = '/home/sondreab/Desktop/msc-file-share/D20191125T125927.756743-inferred.png'
raw = '/home/sondreab/Desktop/msc-file-share/D20191125T125927.756743.png'

raw = cv2.imread(raw)
inferred = cv2.imread(inferred)

raw_bottom_right = raw[raw.shape[0]//2:, raw.shape[1]//2:, :]
inferred_bottom_right = inferred[inferred.shape[0]//2:, inferred.shape[1]//2:, :]

raw_bottom_left = raw[raw.shape[0]//2:, :raw.shape[1]//2, :]
inferred_bottom_left = inferred[inferred.shape[0]//2:, :inferred.shape[1]//2, :]

raw_top_right = raw[:raw.shape[0]//2, raw.shape[1]//2:, :]
inferred_top_right = inferred[:inferred.shape[0]//2:, inferred.shape[1]//2:, :]
 
        
#cv2.imwrite('/home/sondreab/Desktop/msc-file-share/raw-zoom_bottom_left.png', raw_bottom_left)
#cv2.imwrite('/home/sondreab/Desktop/msc-file-share/inferred-zoom_bottom_left.png', inferred_bottom_left)

#cv2.imwrite('/home/sondreab/Desktop/msc-file-share/raw-zoom_top_right.png', raw_top_right)
#cv2.imwrite('/home/sondreab/Desktop/msc-file-share/inferred-zoom_top_right.png', inferred_top_right)
"""


# In[ ]:


def read_json_file(file_name, directory):
    json_file = os.path.join(directory, file_name+'.json')
    with open(json_file) as f:
        data = json.load(f)
    return data


# In[ ]:


def create_json_file(data, file_name, directory):
    json_file = os.path.join(directory, file_name + '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# In[ ]:


def clean_vgg_annotator_coco_file(file_name, directory):
    
    """
    COCO DATA FORMAT:
    
    
    info = {
        "year"          : int, 
        "version"       : str, 
        "description"   : str, 
        "contributor"   : str, 
        "url"           : str, 
        "date_created"  : datetime,
        }
    
    images = [{    
        "id"            : int,
        "width"         : int, 
        "height"        : int, 
        "file_name"     : str, 
        "license"       : int, 
        "flickr_url"    : str, 
        "coco_url"      : str, 
        "date_captured" : datetime,
        }]
    
    annotations = [{
        "id"            : int, 
        "image_id"      : int, 
        "category_id"   : int, 
        "segmentation"  : RLE or [polygon], 
        "area"          : float, 
        "bbox"          : [x,y,width,height], 
        "iscrowd"       : 0 or 1,
        }]
 
    
    licences = [{
        "id"            : int,
        "name"          : str, 
        "url"           : str,
    }]
    
    categories = [{
        "id"            : int, 
        "name"          : str, 
        "supercategory" : str,
        }]
    
    dataset = {
        "info"          : info, 
        "images"        : images,      #[image], 
        "annotations"   : annotations, #[annotation], 
        "licenses"      : licenses,    #[license],
        "categories"    : categories,  #[category],
        }    
    """
    
    coco_labels = read_json_file(file_name, directory)
    
    dataset = coco_labels
    for field_name, field in coco_labels.items():
        
        if field_name == "annotations":
            annotations = []
            for item in field:
                annotation = item
                image_id = item['image_id']
                annotation['image_id'] = int(image_id)
                annotation["category_id"] = 5
                segmentation = annotation["segmentation"].copy()
                annotation["segmentation"] = [segmentation]
                annotations.append(annotation)
            dataset['annotations'] = annotations
            
            
    
    classes = ['oil', 'other', 'bubble', 'faecal_pellets', 'copepod', 'diatom_chain', 'oily_gas']
    categories = []
    for idx, name in enumerate(classes):
        if (name == 'faecal_pellets') or (name == 'copepod') or (name == 'diatom_chain'):
            supercategory = 'plankton'
        else:
            supercategory = 'other'
    
        categories.append(
            {'id': idx+1, 
             'name': name, 
             'supercategory': supercategory,})
        
    dataset['categories'] = categories
    dataset['licenses'] = [{'id': 1, 'name': 'Public Domain', 'url': ''}]
    #print(dataset)
    return_file = file_name+'_clean'
    create_json_file(data=dataset, file_name=return_file, directory='/home/sondreab/Desktop/msc-file-share/json')
    return return_file


# In[ ]:


def extract_sample_list_from_coco_set(file_name, directory):
    coco_labels = read_json_file(file_name, directory)
    
    files = []
    for image in coco_labels['images']:
        files.append(image['file_name'])
    return_file = file_name+'_files'
    create_json_file(files, file_name=return_file, directory=directory)
    return return_file


# In[ ]:


def split_sample_list(file_name, directory):
    file_list = read_json_file(file_name, directory)
    train = []
    val = []
    test = []
    for file in file_list:
        file = ".".join(file.split('.')[:-1])
        if np.random.rand(1) < 0.15:
            val.append(file)
        elif np.random.rand(1) > 0.85:
            test.append(file)
        else:
            train.append(file)
    print('Total: {}, Train: {}, Val: {}, Test: {}'.format(len(file_list), len(train), len(val), len(test)))
    split_sets = {'train': train, 'val': val, 'test': test}
    return_file = file_name+'_split_sets'
    create_json_file(split_sets, file_name=return_file, directory=directory)
    return return_file


# In[ ]:


def split_coco_set(dataset_file_name, split_file_name, directory):
    coco_labels = read_json_file(dataset_file_name, directory)
    set_assignment = read_json_file(split_file_name, directory)
    
    train = {}
    train["info"] = coco_labels["info"]
    train["licenses"] = coco_labels["licenses"]
    train["categories"] = coco_labels["categories"]
    train["images"] = []
    train["annotations"] = []
    
    val = {}
    val["info"] = coco_labels["info"]
    val["licenses"] = coco_labels["licenses"]
    val["categories"] = coco_labels["categories"]
    val["images"] = []
    val["annotations"] = []
    
    test = {}
    test["info"] = coco_labels["info"]
    test["licenses"] = coco_labels["licenses"]
    test["categories"] = coco_labels["categories"]
    test["images"] = []
    test["annotations"] = []
    
    sets = ["train", "val", "test"]
    dicts = [train, val, test]
    
    
    for image in coco_labels["images"]:
        file_name = image["file_name"]
        file_name = ".".join(file_name.split('.')[:-1])
        for idx, set_ in enumerate(sets):
            if file_name in set_assignment[set_]:
                #print("appending")
                dicts[idx]["images"].append(image)
                for annotation in coco_labels["annotations"]:
                    if annotation["image_id"] == image["id"]:
                        dicts[idx]["annotations"].append(annotation)
    return_files = [dataset_file_name+'_',dataset_file_name+'_', dataset_file_name+'_']
    for idx, set_ in enumerate(sets):
        return_files[idx] += sets[idx]
        create_json_file(dicts[idx], file_name=return_files[idx], directory=directory)
    return return_files


# In[ ]:


def from_coco_to_detectron_default_dataset(coco_dataset, coco_directory, image_directory):
    coco_labels = read_json_file(coco_dataset, coco_directory)
    dataset = []
    for image in coco_labels["images"]:
        record = {}
        record["file_name"] = os.path.join(image_directory, image["file_name"])
        record["image_id"] = image["id"]
        record["height"] = image["height"]
        record["width"] = image["width"]
        annotations = []
        for annotation in coco_labels["annotations"]:
            ann = {}
            if annotation["image_id"] == record["image_id"]:
                xmin = annotation["bbox"][0]
                ymin = annotation["bbox"][1]
                xmax = xmin + annotation["bbox"][2]
                ymax = ymin + annotation["bbox"][3]
                ann["bbox"] = [xmin, ymin, xmax, ymax]
                ann["segmentation"] = annotation["segmentation"]
                ann["category_id"] = annotation["category_id"] - 1
                ann["iscrowd"] = 0
                annotations.append(ann.copy())
                #coco_labels["annotations"].remove(annotation)
        record["annotations"] = annotations
        dataset.append(record.copy())
    create_json_file(dataset, file_name = coco_dataset+"_default_detectron", directory=coco_directory)
        
            
        


# In[ ]:


def from_detectron_to_coco_default_dataset(detectron_dataset, dataset_directory, area_threshold = 0):
    detectron_labels = read_json_file(detectron_dataset, dataset_directory)
    image_ids = read_json_file("image_id_mapping", "/home/sondreab/Desktop/msc-file-share/json")
    coco_labels = {}
    coco_labels["info"] = {
        "year": 2020,
        "version": "1",
        "description": "Coco format dataset",
        "contributor": "Sondre Bergum",
        "url": "https://github.com/AILARON/Segmentation/"
    }
    coco_labels["categories"] = [
        {
            "id": 1,
            "name": "oil",
            "supercategory": "other"
        },
        {
            "id": 2,
            "name": "other",
            "supercategory": "other"
        },
        {
            "id": 3,
            "name": "bubble",
            "supercategory": "other"
        },
        {
            "id": 4,
            "name": "faecal_pellets",
            "supercategory": "plankton"
        },
        {
            "id": 5,
            "name": "copepod",
            "supercategory": "plankton"
        },
        {
            "id": 6,
            "name": "diatom_chain",
            "supercategory": "plankton"
        },
        {
            "id": 7,
            "name": "oily_gas",
            "supercategory": "other"
        }
    ]
    coco_labels["licenses"] = [
        {
            "id":1,
            "name":"GPLv3 - GNU General Public License",
            "url":"https://www.gnu.org/licenses/gpl-3.0.html"
        }
    ]
    
    images = []
    annotations = []
    
    object_count = 0
    
    for image in detectron_labels:
        file_name = '.'.join(image["file_name"].split('/')[-1].split('.')[:-1])
        image_id = image_ids[file_name]
        coco_image = {
            "id": image_id,
            "width": image["width"],
            "height": image["height"],
            "file_name": file_name + ".bmp",
            "license": 1,
            "date_captured": ""
        }
        
        images.append(coco_image.copy())
        for annotation in image["annotations"]:
            bbox = annotation["bbox"]
            bbw = bbox[2]-bbox[0]
            bbh = bbox[3]-bbox[1]
            area = bbw*bbh
            if area < area_threshold:
                continue
            coco_annotation = {
                "id": object_count,
                "image_id": image_id,
                "segmentation": annotation["segmentation"],
                "area": area,
                "bbox": [bbox[0], bbox[1], bbw, bbh],
                "iscrowd": 0,
                "category_id": annotation["category_id"]+1                
            }
            
            annotations.append(coco_annotation.copy())
            object_count += 1
    coco_labels["images"] = images
    coco_labels["annotations"] = annotations
    return_file = detectron_dataset+"_coco"
    create_json_file(coco_labels, file_name = return_file, directory=dataset_directory)
    return return_file
    


# In[ ]:


from_detectron_to_coco_default_dataset(detectron_dataset="copepod_stats",
                                       dataset_directory= "/home/sondreab/Desktop/msc-file-share/json")


# In[ ]:


from_detectron_to_coco_default_dataset(detectron_dataset="copepod_stats",
                                       dataset_directory= "/home/sondreab/Desktop/msc-file-share/json",
                                       area_threshold = 32*32)


# In[ ]:


def raw_to_imc_filenames_coco(file_name, directory):
    coco_labels = read_json_file(file_name, directory)
    
    for image in coco_labels["images"]:
        image_name = image["file_name"]
        split_name = image_name.split('.')
        new_image_name = '.'.join(split_name[:-1])+'-IMC.'+split_name[-1]
        image["file_name"] = new_image_name
    create_json_file(coco_labels, file_name = file_name+"_IMC", directory=directory)
    
def raw_to_imc_filenames(file_name, directory):
    detectron_labels = read_json_file(file_name, directory)
    
    for image, field in detectron_labels.items():
        image_name = field["filename"]
        split_name = image_name.split('.')
        new_image_name = '.'.join(split_name[:-1])+'-IMC.'+split_name[-1]
        image["file_name"] = new_image_name
    create_json_file(detectron_labels, file_name = file_name+"_IMC", directory=directory)


# In[ ]:


raw_to_imc_filenames_coco("via_export_coco", '/home/sondreab/Desktop/msc-file-share/json')
raw_to_imc_filenames("via_export_json", '/home/sondreab/Desktop/msc-file-share/json')


# In[ ]:


image_name = "D20191125T125406.862891.bmp"
print(image_name.split('.')[:-1])
split_name = image_name.split('.')
print('.'.join(split_name[:-1])+'-IMC.'+split_name[-1])
#new_image_name = '.'.join(image_name.


# In[ ]:


dictionary = read_json_file("complete_manual_copepod_annotations", '/home/sondreab/Desktop/msc-file-share/json')
for annotation in dictionary["annotations"]:
                segmentation = annotation["segmentation"].copy()
                print(segmentation)
                annotation["segmentation"] = [segmentation]
                print(annotation["segmentation"])
create_json_file(dictionary, file_name="complete_manual_copepod_annotations_wrapped", directory='/home/sondreab/Desktop/msc-file-share/json')


# In[ ]:


file = "complete_manual_copepod_annotations"
dictionary = read_json_file(file, '/home/sondreab/Desktop/msc-file-share/json')
mapping = {}
for image in dictionary["images"]:
    image_name = '.'.join(image["file_name"].split('.')[:-1])
    mapping[image_name] = image["id"]
create_json_file(mapping, 
                 file_name=file+"_mapping", 
                 directory='/home/sondreab/Desktop/msc-file-share/json')


# In[ ]:


###### Correcting category ID in the raw export from VIA

file = "via_export_json"
dictionary = read_json_file(file, '/home/sondreab/Desktop/msc-file-share/json')

for image, fields in dictionary.items():
    print(image)
    #print(fields)#['regions'])#[0]['region_attributes'])
    for region in fields['regions']:
        region['region_attributes']['category_id'] = "4"
create_json_file(dictionary, file_name="via_export_json_new", directory='/home/sondreab/Desktop/msc-file-share/json')


# In[ ]:


cleaned_file = clean_vgg_annotator_coco_file('via_export_coco', '/home/sondreab/Desktop/msc-file-share/json')


# In[ ]:


extract_sample_list_from_coco_set('complete_manual_coco_test','/home/sondreab/Desktop/msc-file-share/json/test')


# In[ ]:


split_sample_list('complete_manual_coco_test_files','/home/sondreab/Desktop/msc-file-share/json/test')


# In[ ]:


split = read_json_file('complete_manual_coco_test_files_split_sets', '/home/sondreab/Desktop/msc-file-share/json/test')


# In[ ]:


dataset_files = split_coco_set("via_export_coco_clean", 
                               "copepod_train_val_test", 
                               "/home/sondreab/Desktop/msc-file-share/json/")


# In[ ]:


dataset_files = split_coco_set("coco_clean", 
                               "copepod_train_val_test", 
                               "/home/sondreab/Desktop/msc-file-share/json/")


# In[ ]:


dataset_files = split_coco_set("copepod_stats_coco_size",
                               "copepod_train_val_test", 
                               "/home/sondreab/Desktop/msc-file-share/json/")


# In[ ]:


for set_ in dataset_files:
    from_coco_to_detectron_default_dataset(coco_dataset = set_,
                                       coco_directory = "/home/sondreab/Desktop/msc-file-share/json/", 
                                       image_directory = '/home/sondreab/Desktop/DATA/copepod_lab_petridish/copepods')

