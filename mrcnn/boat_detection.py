import os
import sys
import random
import math
import numpy as np
import skimage.io
import pickle
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
# import mrcnn.model as modellib
# from mrcnn import visualize
import model as modellib
import visualize
import config as Configure


def group(lst1, n):
    for i in range(0, len(lst1), n):
        val = lst1[i:i+n]
        if len(val) == n:
            yield tuple(val)

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = "/kaggle/input/model/mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
# IMAGE_DIR = '/kaggle/input/airbus-ship-detection/test/'

class InferenceConfig(Configure.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    NUM_CLASSES = 81

config = InferenceConfig()
batch_size = 15
config.IMAGES_PER_GPU = batch_size
config.BATCH_SIZE = config.IMAGES_PER_GPU * config.GPU_COUNT
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

TEST_PATH = '/kaggle/input/airbus-ship-detection/test/'
im_names = os.listdir(TEST_PATH)
data = list(group(im_names, batch_size))
# leftover = im_names[len(im_names)-6:]

START = 0
END = len(data)

all_results = []
for idx, d in enumerate(data[START:END]):
    print('Processing batch {}...'.format(START+idx))
    im_batch = []
    for im in d:
        im_batch.append(skimage.io.imread(os.path.join(TEST_PATH, im)))
    results = model.detect(im_batch, verbose=0)
    for i in range(len(results)):
        results[i]['im_name'] = d[i]
    all_results += results
    with open('output.txt', 'w') as f:
    	f.write('Processing batch {}/{}...'.format(START+idx, len(data)))
    with open('results/results_{}_{}.pkl'.format(START, END), 'wb') as f:
        pickle.dump(all_results, f)
