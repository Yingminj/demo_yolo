'''
Author: abner
Date: 2025-10-16 16:06:01
LastEditTime: 2025-10-17 10:37:21
Description: 
FilePath: /Demo_1016/main.py
'''

from camera_test.camera_base import Camera

import cv2
import datetime
import os
import time
from ultralytics import YOLO

if __name__ == "__main__":
    camera = Camera(device="/dev/video0", fps=60)
    yolo_model = YOLO("weight/best.pt")
    if not os.path.exists("cap"):
        os.makedirs("cap")

    # frame = camera.get_frame()    
    for frame in camera.get_frame():
        results = yolo_model.track(frame, persist=True, iou = 0.3, tracker="bytetrack.yaml")[0] #yolo track

        grid_status = ['free'] * 24  # 24 grids, default free

        # Process detections
        if results.boxes is not None:
            for box in results.boxes:
                cls = int(box.cls.item())
                class_name = results.names[cls]
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]

                if class_name == 'sorting_tray':
                    # Tray dimensions: 0.55m x 0.35m, grids: 6x4 = 24, each 0.085m x 0.09m approx
                    tray_width = bbox[2] - bbox[0]
                    tray_height = bbox[3] - bbox[1]
                    grid_width = tray_width / 6
                    grid_height = tray_height / 4

                    # Draw grid on frame
                    for i in range(7):  # vertical lines
                        x = int(bbox[0] + i * grid_width)
                        cv2.line(frame, (x, int(bbox[1])), (x, int(bbox[3])), (255, 0, 0), 1)
                    for j in range(5):  # horizontal lines
                        y = int(bbox[1] + j * grid_height)
                        cv2.line(frame, (int(bbox[0]), y), (int(bbox[2]), y), (255, 0, 0), 1)

                elif class_name in ['cola', 'mug']:
                    # Get center
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2

                    # Check if tray exists and find grid
                    if 'sorting_tray' in [results.names[int(b.cls.item())] for b in results.boxes]:
                        tray_bbox = None
                        for b in results.boxes:
                            if results.names[int(b.cls.item())] == 'sorting_tray':
                                tray_bbox = b.xyxy[0].cpu().numpy()
                                break
                        if tray_bbox is not None:
                            tray_width = tray_bbox[2] - tray_bbox[0]
                            tray_height = tray_bbox[3] - tray_bbox[1]
                            grid_width = tray_width / 6
                            grid_height = tray_height / 4

                            # Find grid index
                            rel_x = center_x - tray_bbox[0]
                            rel_y = center_y - tray_bbox[1]
                            if 0 <= rel_x < tray_width and 0 <= rel_y < tray_height:
                                col = int(rel_x // grid_width)
                                row = int(rel_y // grid_height)
                                idx = row * 6 + col
                                if 0 <= idx < 24:
                                    grid_status[idx] = 'occ'

        # Visualize grid status on frame
        if 'sorting_tray' in [results.names[int(b.cls.item())] for b in results.boxes if results.boxes is not None]:
            tray_bbox = None
            for b in results.boxes:
                if results.names[int(b.cls.item())] == 'sorting_tray':
                    tray_bbox = b.xyxy[0].cpu().numpy()
                    break
            if tray_bbox is not None:
                tray_width = tray_bbox[2] - tray_bbox[0]
                tray_height = tray_bbox[3] - tray_bbox[1]
                grid_width = tray_width / 6
                grid_height = tray_height / 4
                for idx in range(24):
                    row = idx // 6
                    col = idx % 6
                    x1 = int(tray_bbox[0] + col * grid_width)
                    y1 = int(tray_bbox[1] + row * grid_height)
                    x2 = int(x1 + grid_width)
                    y2 = int(y1 + grid_height)
                    color = (0, 255, 0) if grid_status[idx] == 'free' else (0, 0, 255)  # green for free, red for occ
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, str(idx), (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # annotated_frame = results.plot()
        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.cleanup()