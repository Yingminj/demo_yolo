from camera_test.camera_base import Camera

import cv2
import datetime
import os
import time
from ultralytics import YOLO
import numpy as np
from scipy.spatial import ConvexHull
from scene_process.table_grid import process_yolo_results
from vis.vis import visualize_grid_with_mask
import threading
import queue

if __name__ == "__main__":
    camera_head = Camera(device="/dev/video2", fps=60, width=640, height= 480, undistortion=True)
    camera_top = Camera(device="/dev/video0", fps=60)

    if not os.path.exists("cap"):
        os.makedirs("cap")

    # 创建队列来存储帧
    frame_queue_head = queue.Queue(maxsize=10) ##速度？
    frame_queue_top = queue.Queue(maxsize=10)

    # 线程函数：捕获帧并放入队列
    def capture_frames(camera, queue):
        for frame in camera.get_frame():
            if not queue.full():
                queue.put(frame)
            else:
                # 如果队列满，丢弃旧帧或处理
                pass

    # 启动线程
    thread_head = threading.Thread(target=capture_frames, args=(camera_head, frame_queue_head))
    thread_top = threading.Thread(target=capture_frames, args=(camera_top, frame_queue_top))
    thread_head.start()
    thread_top.start()

    # 主循环：同时处理两个相机的帧
    while True:
        try:
            frame_head = frame_queue_head.get(timeout=1)  # 超时避免阻塞
            frame_top = frame_queue_top.get(timeout=1)
            # left_image = frame_head[:, 640:, :]
            # 处理帧，例如显示或进一步处理
            cv2.imshow('camera_head', frame_head)
            cv2.imshow('camera_top', frame_top)
            
            # 添加你的 YOLO 处理逻辑在这里，例如：
            # results_head = model(frame_head)
            # results_top = model(frame_top)
            # process_yolo_results(results_head, results_top)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            continue  # 如果队列空，继续等待

    # 清理
    camera_head.cleanup()
    camera_top.cleanup()
    cv2.destroyAllWindows()