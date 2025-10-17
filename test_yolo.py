'''
Author: abner
Date: 2025-10-16 16:06:01
LastEditTime: 2025-10-17 14:24:25
Description: 
FilePath: /Demo_1016/test_yolo.py
'''

from camera_test.camera_base import Camera

import cv2
import datetime
import os
import time
from ultralytics import YOLO

if __name__ == "__main__":
    camera = Camera(device="/dev/video0", fps=60)
    yolo_model = YOLO("weight/best_seg.pt")
    if not os.path.exists("cap"):
        os.makedirs("cap")

    # frame = camera.get_frame()    
    for frame in camera.get_frame():
        results = yolo_model.track(frame, persist=True, iou = 0.3, tracker="bytetrack.yaml")[0] #yolo track
        annotated_frame = results.plot()
        cv2.imshow('YOLO Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.cleanup()