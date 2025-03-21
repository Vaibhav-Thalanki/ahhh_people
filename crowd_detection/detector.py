import torch
import cv2
import numpy as np
from ultralytics import YOLO

class CrowdDetector:
    def __init__(self, model_path="yolov8x.pt", conf_threshold=0.3):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_people(self, image):
        results = self.model.predict(source=image, conf=self.conf_threshold, classes=[0])
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        return boxes
