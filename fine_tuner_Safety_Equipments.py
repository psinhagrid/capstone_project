from ultralytics import YOLO
import gc
import os
import torch


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    #print (x)
else:
    print ("MPS device not found.")

import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
from fastai.vision.all import *

model = YOLO('venv/YOLO-weights/yolov8l.pt')

# Train the model with 2 GPUs
results = model.train(data='/Users/psinha/Documents/capstone_project/venv/YOLO-fine_tuner/fine_tuning_data/data.yaml', epochs=2, batch=4, imgsz=(640, 480), device='mps')