import os

from ultralytics import YOLO
# Load a model
#model = YOLO("runs/detect/train2/weights/best.pt")  # build a new model from scratch
#model = YOLO("runs/detect/yolov5/weights/best.pt")  # build a new model from scratch train6是scratch训练的权重
model=YOLO('yolo11l.pt')
# Use the model
train_results = model.train(data="data/data1.yaml", epochs=100, imgsz=640,  device=0,)
