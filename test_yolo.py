from camera_test.camera_base import Camera

import cv2
import datetime
import os
import time
from ultralytics import YOLO

if __name__ == "__main__":
    camera = Camera(device="/dev/video0", fps=60)
    yolo_model = YOLO("/home/kewei/YING/yolo_cube/runs/segment/train7/weights/best.pt")
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