import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import coco
import utils
import visualize
from config import Config
from visualize import display_images
import model as modellib
from model import log
from supported_architectures import Architecture
#matplotlib inline

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "log")
config = coco.CocoConfig()
COCO_DIR = "/root/data/MOBILE_MASKRCNN/images"  # TODO: enter your own path here

#configuration
class CocoConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    # NUM_CLASSES = 1 + 80  # COCO has 80 classes
    NUM_CLASSES = 1 + 1  # Person and background

    # Which architecture type

    STEPS_PER_EPOCH = 1000
    NUM_CLASSES = 1 + 1  # Person and background

    # Which architecture type
    BACKBONE = "mobilenetv1"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    ## Losses
    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 0.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    TRAIN_BN = False
    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    NUM_KEYPOINTS = 17 #Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
config = coco.CocoConfig()
config.display()

# Load dataset
assert config.NAME == "coco"
# Training dataset
# load person keypoints dataset
train_dataset_keypoints = coco.CocoDataset(task_type="person_keypoints")
train_dataset_keypoints.load_coco(COCO_DIR, "train")
train_dataset_keypoints.prepare()

#Validation dataset
val_dataset_keypoints = coco.CocoDataset(task_type="person_keypoints")
val_dataset_keypoints.load_coco(COCO_DIR, "val")
val_dataset_keypoints.prepare()

print("Train Keypoints Image Count: {}".format(len(train_dataset_keypoints.image_ids)))
print("Train Keypoints Class Count: {}".format(train_dataset_keypoints.num_classes))
for i, info in enumerate(train_dataset_keypoints.class_info):
    print("{:3}. {:50}".format(i, info['name']))

print("Val Keypoints Image Count: {}".format(len(val_dataset_keypoints.image_ids)))
print("Val Keypoints Class Count: {}".format(val_dataset_keypoints.num_classes))
for i, info in enumerate(val_dataset_keypoints.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
#init_with = "last"
#model_path = model.find_last()[1]
# Load weights
#print("Loading weights ", model_path)
# Load the last model you trained and continue training
#model.load_weights(model_path, by_name=True)

# Training - Stage 1
# Finetune layers from mobilenet_v1 stage 3 and up
print("Training all layers")
model.train(train_dataset_keypoints, val_dataset_keypoints,
            learning_rate=config.LEARNING_RATE,
            epochs=200,
            layers='all')
#save model
model_path = os.path.join(MODEL_DIR, "mobile_mask_rcnn_coco_humanpose.h5")
model.keras_model.save_weights(model_path)
