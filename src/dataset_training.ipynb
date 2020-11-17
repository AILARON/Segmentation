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
# DEPENDENT ON:
# https://github.com/facebookresearch/detectron2

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


# In[ ]:


from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode
from time import gmtime, strftime


DIRECTORY = '/home/sondreab/Desktop/DATA/copepod_lab_petridish'


cfg = get_cfg()

runtime = strftime("%Y.%m.%d_%H:%M:%S", gmtime())

DATA_DIR = DIRECTORY + '/copepods'
VISUALIZE_DIR = DIRECTORY + '/visualize/'+ runtime
INFERENCE_DIR = DIRECTORY + '/inference/' + runtime

DATASET = 'copepods'

OUTPUT_PATH = DIRECTORY + "/output/" + "model_" + runtime
cfg.OUTPUT_DIR = OUTPUT_PATH
#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(OUTPUT_PATH)


# In[ ]:


os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[ ]:

#########################################################################################
#Code in cell is from https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
#########################################################################################

# In[ ]:

#########################################################################################
#Code in cell is from https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
                     
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks
#########################################################################################

# In[ ]:


def create_json_file(data, file_name, directory=DIRECTORY):
    json_file = os.path.join(directory, file_name + '.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# In[ ]:


def read_json_file(file_name, directory=DIRECTORY):
    json_file = os.path.join(directory, file_name+'.json')
    with open(json_file) as f:
        dataset = json.load(f)
    for record in dataset:
        for obj in record['annotations']:
            obj["bbox_mode"] = BoxMode.XYXY_ABS
    return dataset


# In[ ]:


def save_dataset_visualization(dataset, directory=VISUALIZE_DIR):
    savepath = VISUALIZE_DIR
    print("Savepath: {}".format(savepath))
    os.makedirs(savepath, exist_ok=True)
    dataset_metadata = MetadataCatalog.get(dataset)
    dataset_dicts = read_json_file(dataset, DIRECTORY)
    print('Saving dataset '+ dataset)
    for image in dataset_dicts:
        img = cv2.imread(image["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(image)
        
        cv2.imwrite(os.path.join(savepath, '.'.join(image['file_name'].split('/')[-1].split('.')[:-1]) + '.png'), vis.get_image()[:, :, ::-1])
        print('.'.join(image['file_name'].split('/')[-1].split('.')[:-1])+ ' saved!')


# In[ ]:


def save_coco_dataset_visualization(dataset,dataset_dir = OUTPUT_PATH, directory=VISUALIZE_DIR):
    savepath = VISUALIZE_DIR
    print("Savepath: {}".format(savepath))
    os.makedirs(savepath, exist_ok=True)
    dataset_metadata = MetadataCatalog.get(dataset)
    dataset_dicts = load_coco(dataset, DIRECTORY)
    print('Saving dataset '+ dataset)
    for image in dataset_dicts:
        img = cv2.imread(image["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(image)
        
        cv2.imwrite(os.path.join(savepath, '.'.join(image['file_name'].split('/')[-1].split('.')[:-1]) + '.png'), vis.get_image()[:, :, ::-1])
        print('.'.join(image['file_name'].split('/')[-1].split('.')[:-1])+ ' saved!')


# In[ ]:


def register_dataset(directory = DIRECTORY):
    thing_classes = ['oil', 'other', 'bubble', 'faremoteecal_pellets', 'copepod', 'diatom_chain', 'oily_gas']
    for d in ["train", "val"]:
        DatasetCatalog.register("copepod_" + d, lambda d=d: read_json_file(d, DIRECTORY))
        MetadataCatalog.get("copepod_" + d).set(thing_classes=thing_classes)


# In[ ]:

#DEPRECATED
def train_dataset(dataset):
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #cfg.merge_from_list(["MODEL.WEIGHTS", os.path.join(DIRECTORY, "output/model_final.pth")])
    cfg.DATASETS.TRAIN = (dataset,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  # only has one class (ballon)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()


# In[ ]:


def inference_over_directory(dataset,
                             files_dir,
                             output_path = OUTPUT_PATH, 
                             inference_dir = os.path.join(DIRECTORY, INFERENCE_DIR)):
    mypath = files_dir
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_PATH, "model_final.pth")
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model   
    
    cfg.DATASETS.TEST = (dataset, )

    predictor = DefaultPredictor(cfg)

    savepath = inference_dir
    os.makedirs(savepath, exist_ok=True)
    
    dataset_metadata = MetadataCatalog.get(dataset)
    
    image_list = onlyfiles
    
    for image in image_list:
        print(os.path.join(mypath,image))
        im = cv2.imread(os.path.join(mypath,image))
        outputs = predictor(im)
        
        vis = Visualizer(im[:, :, ::-1],
                    metadata=dataset_metadata, 
                    scale=0.5, 
                    instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
            )
        v = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(savepath, '.'.join(image.split('.')[:-1]) + '.png'), v.get_image()[:, :, ::-1])
    print(savepath)


# In[ ]:


sets = ["train",  "val", "test"]
thing_classes = ['oil', 'other', 'bubble', 'faecal_pellets', 'copepod', 'diatom_chain', 'oily_gas']
for set_ in sets:
    dataset = "my_dataset_"+set_
    file = "my_dataset_"+set_
    DatasetCatalog.register(dataset, lambda d=file: read_json_file(d, DIRECTORY))
    MetadataCatalog.get(dataset).set(thing_classes=thing_classes)


# In[ ]:


from detectron2.data.datasets import register_coco_instances
sets = ["train",  "val", "test"]
for d in sets:
    register_coco_instances("my_coco_dataset_"+d, 
                            {}, 
                            "/home/sondreab/Desktop/DATA/copepod_lab_petridish/my_coco_dataset_"+d+".json", 
                            "/home/sondreab/Desktop/DATA/copepod_lab_petridish/copepods")
    
register_coco_instances("copepod_stats_coco_train", 
                            {}, 
                            "/home/sondreab/Desktop/DATA/copepod_lab_petridish/copepod_stats_coco_train.json", 
                            "/home/sondreab/Desktop/DATA/copepod_lab_petridish/copepods")
register_coco_instances("copepod_stats_coco_test", 
                            {}, 
                            "/home/sondreab/Desktop/DATA/copepod_lab_petridish/copepod_stats_coco_test.json", 
                            "/home/sondreab/Desktop/DATA/copepod_lab_petridish/copepods")


# In[ ]:


########### Setup backbone chekcpoint

task = "COCO-SemanticSegmentation"
arch_backbone = "mask_rcnn_X_101_32x8d_FPN_3x"

cfg = get_cfg()

#runtime = strftime("%Y.%m.%d_%H:%M:%S", gmtime())

OUTPUT_PATH = DIRECTORY + "/output/" + "model_" + arch_backbone + "_" + runtime
cfg.OUTPUT_DIR = OUTPUT_PATH
#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(OUTPUT_PATH)


# In[ ]:



cfg.merge_from_file(model_zoo.get_config_file(task + "/" + arch_backbone + ".yaml"))
cfg.DATASETS.TRAIN = ("copepod_stats_coco_train",)
cfg.DATASETS.TEST = ("my_coco_dataset_val",)
cfg.TEST.EVAL_PERIOD = 1000
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(task + "/" + arch_backbone + ".yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  
cfg.SOLVER.MAX_ITER = 20000  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
#cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 5   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7
#cfg.MODEL.RETINANET.NUM_CLASSES = 7

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

AP_EVAL_METRICS = {}

evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir=OUTPUT_PATH)
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
detectron_metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
AP_EVAL_METRICS["detectron"] = detectron_metrics


evaluator = COCOEvaluator("my_coco_dataset_test", cfg, False, output_dir=OUTPUT_PATH)
val_loader = build_detection_test_loader(cfg, "my_coco_dataset_test")
coco_metrics = inference_on_dataset(trainer.model, val_loader, evaluator)
AP_EVAL_METRICS["coco"] = coco_metrics

create_json_file(AP_EVAL_METRICS, "AP_eval_metrics", OUTPUT_PATH)


# In[ ]:


def inference(dataset, 
              output_path = OUTPUT_PATH, 
              inference_dir = os.path.join(DIRECTORY, INFERENCE_DIR), 
              weights = "model_final.pth"):
    
    
    cfg.MODEL.WEIGHTS = os.path.join(output_path, weights)
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set the testing threshold for this model   
    
    cfg.DATASETS.TEST = (dataset, )
    
    predictor = DefaultPredictor(cfg)

    savepath = inference_dir
    os.makedirs(savepath, exist_ok=True)
    
    dataset_metadata = MetadataCatalog.get(dataset)
    dataset_dicts = read_json_file(dataset, DIRECTORY)
    
    for image in dataset_dicts:    
        im = cv2.imread(image["file_name"])
        outputs = predictor(im)
        #print(outputs)
        #create_json_file(outputs, 'outputs')
        
        vis = Visualizer(im[:, :, ::-1],
                   metadata=dataset_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE  # remove the colors of unsegmented pixels
        )
        v = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
        #cv2.imshow('prediction',v.get_image()[:, :, ::-1])
        #cv2.waitKey(0)
        cv2.imwrite(os.path.join(savepath, '.'.join(image['file_name'].split('/')[-1].split('.')[:-1]) + '.png'), v.get_image()[:, :, ::-1])
    print(savepath)


# In[ ]:


inference("my_dataset_test")


# In[ ]:


#save_dataset_visualization("my_dataset_train", VISUALIZE_DIR)
#save_dataset_visualization("my_dataset_val", VISUALIZE_DIR)
save_dataset_visualization("my_dataset_test", VISUALIZE_DIR)


# In[ ]:


save_dataset_visualization("my_coco_dataset_train", VISUALIZE_DIR)
save_dataset_visualization("my_coco_dataset_val", VISUALIZE_DIR)
save_dataset_visualization("my_coco_dataset_test", VISUALIZE_DIR)


# In[ ]:


inference_over_directory(dataset= "my_dataset_test",
                         files_dir = os.path.join(DIRECTORY,"RAWbmp"),
                         output_path = OUTPUT_PATH, 
                         inference_dir = os.path.join(DIRECTORY, INFERENCE_DIR)+"_mission_test")

