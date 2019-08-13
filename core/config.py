#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:06:54
#   Description :
#
#================================================================

from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.CLASSES                = "./data/classes/classes_5.names"
#"./data/classes/coco.names"
__C.YOLO.ANCHORS                = "./data/anchors/basline_anchors.txt"#coco_anchors.txt"#basline_anchors.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5 #.45 #used to be .5
__C.YOLO.UPSAMPLE_METHOD        = "resize"
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt"#yolov3_test_loss=20.7890.ckpt-old"#yolov3_coco_demo.ckpt"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.ANNOT_PATH            = "./data/dataset/train_5.txt" #flickr_train_big.txt"
#"./data/dataset/voc_train.txt"
__C.TRAIN.BATCH_SIZE            = 2 #6 triggered OOM exception
#6
__C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736]  # only used to be up to size 608
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 2e-6 #2e-6 #5e-5 #1e-4 #1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-7 #1e-7 #5e-7 # 1e-6  
__C.TRAIN.WARMUP_EPOCHS         = 2 # was 2
__C.TRAIN.FIRST_STAGE_EPOCHS    = 20 # was 20  # tried 10/100
__C.TRAIN.SECOND_STAGE_EPOCHS   = 160 #160 # was 30
__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_coco_demo.ckpt"#yolov3_coco_demo.ckpt"#yolov3_test_loss=20.7890.ckpt-old"#yolov3_coco_demo.ckpt"
__C.TRAIN.FREEZE_OUTPUT         = "./yolov3_frozen_5.pb"



# TEST options
__C.TEST                        = edict()

__C.TEST.ANNOT_PATH             = "./data/dataset/test_5.txt" #flickr_test_big.txt"
#"./data/dataset/voc_test.txt"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.INPUT_SIZE             = 544
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_test_loss=28.8257.ckpt-80" # yolov3_test_loss=2.1426.ckpt-lre_1e-7" #yolov3_test_loss=2.9426.ckpt-save1"
#"./checkpoint/yolov3_test_loss=9.2099.ckpt-5"
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3 #01
__C.TEST.IOU_THRESHOLD          = 0.45 # originally 0.5 - not relevant for our pipeline, can be used by evaluate.py
