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

import json
import os
import matplotlib.pyplot as plt

METRICS_folder = '/home/sondreab/Desktop/METRICS'


def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def load_files_in_dir(directory = "./"):
    mypath = directory
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    files = []
    for file_name in onlyfiles:
        files.append('_'.join(file_name.split('_')[2:]).split('.')[0])
    return files,onlyfiles


# In[ ]:


models,files = load_files_in_dir(METRICS_folder)
models = ['R_50_C4_1x',
          'R_50_DC5_1x',
          'R_50_FPN_1x',
          'R_50_C4_3x', 
          'R_50_DC5_3x',
          'R_50_FPN_3x', 
          'R_101_C4_3x', 
          'R_101_DC5_3x',
          'R_101_FPN_3x',
          'X_101_32x8d_FPN_3x']
files = ['mask_rcnn_R_50_C4_1x.json',
         'mask_rcnn_R_50_DC5_1x.json',
         'mask_rcnn_R_50_FPN_1x.json', 
         'mask_rcnn_R_50_C4_3x.json',
         'mask_rcnn_R_50_DC5_3x.json',
         'mask_rcnn_R_50_FPN_3x.json', 
         'mask_rcnn_R_101_C4_3x.json',
         'mask_rcnn_R_101_DC5_3x.json',
         'mask_rcnn_R_101_FPN_3x.json', 
         'mask_rcnn_X_101_32x8d_FPN_3x.json']

print(models)
print(files)
example = load_json_arr(os.path.join(METRICS_folder,'mask_rcnn_X_101_32x8d_FPN_3x.json'))
for line in example:
    print(line.items().items())


# In[ ]:


dicts = []

for file, model in zip(files, models) :  
    metrics = load_json_arr(os.path.join(METRICS_folder,file))
    dicts.append(metrics)
    #plt.plot(models,[x['bbox/AP'] for x in metrics if metric in x])
print(dicts[][-1]["time"])


# In[ ]:


##########PLOT
def plot_all_models(models = [],
                    files = [],
                    saveto = "/home/sondreab/Desktop/METRICS/test",
                    figname = 'fig.pdf',
                    metric = "validation loss"):
    dicts = []
    for file, model in zip(files, models) :
        
        metrics = load_json_arr(os.path.join(METRICS_folder,file))
        dicts.append(metrics)
        
        plt.plot(
            [x['iteration'] for x in metrics if metric in x], 
            [x['bbox/AP'] for x in metrics if metric in x])
        plt.legend(models, loc='lower right',  prop={'size': 8})
        plt.xlabel('iteration')
        plt.ylabel(metric)
        #plt.legend(['total_loss', 'validation_loss'], loc='upper right')
        #plt.show()
        
        os.makedirs(saveto, exist_ok=True)
        #print(figname)
        
        #plt.clf
    plt.savefig(os.path.join(saveto,figname))
    
plot_all_models(models = models,
            files = files,
            saveto = "/home/sondreab/Desktop/METRICS/test",
            figname = 'BBOX_AP.pdf',
            metric = "bbox/AP")
