from camera_test.camera_base import Camera
from camera_test.load_cam_params import get_all_camera_params

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
from scipy.spatial.transform import Rotation

def capture_frames(camera, queue):
    for frame in camera.get_frame():
        if not queue.full():
            queue.put(frame)
        else:
            pass # 如果队列满，丢弃旧帧或处理

def project_mask_to_head(mask_points, depth, top_matrix, head_matrix, trans, rot):
    """
    将top相机中的2D掩码点投影到head相机图像上，假设深度为depth。
    """
    if mask_points is None or len(mask_points) == 0:
        return None
    
    # 反投影到3D (假设在Z=depth平面上)
    points_2d = np.array(mask_points, dtype=np.float32)
    undistorted = cv2.undistortPoints(points_2d.reshape(-1, 1, 2), top_matrix, None)
    points_3d = np.hstack([undistorted.reshape(-1, 2), np.full((undistorted.shape[0], 1), depth)])
    
    # 应用外参变换 (top到head)
    points_3d_transformed = (rot @ points_3d.T + trans.reshape(3, 1)).T
    
    # 投影到head图像平面
    projected, _ = cv2.projectPoints(points_3d_transformed, np.eye(3), np.zeros(3), head_matrix, None)
    return projected.reshape(-1, 2)

def project_bbox_to_head(bbox, depth, top_matrix, head_matrix, trans, rot):
    """
    将top相机中的2D bbox投影到head相机图像上，假设深度为depth。
    bbox: [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = bbox
    # 四个角点
    points_2d = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    undistorted = cv2.undistortPoints(points_2d.reshape(-1, 1, 2), top_matrix, None)
    points_3d = np.hstack([undistorted.reshape(-1, 2), np.full((undistorted.shape[0], 1), depth)])
    
    # 应用外参变换 (top到head)
    points_3d_transformed = (rot @ points_3d.T + trans.reshape(3, 1)).T
    
    # 投影到head图像平面
    projected, _ = cv2.projectPoints(points_3d_transformed, np.eye(3), np.zeros(3), head_matrix, None)
    # projected, _ = cv2.projectPoints(points_3d, rot, trans, head_matrix, None)
    return projected.reshape(-1, 2)

if __name__ == "__main__":
    # 初始化相机和模型
    camera_head = Camera(device="/dev/video0", fps=60, width=1280, height= 480, undistortion=True)
    camera_top = Camera(device="/dev/video2", fps=60)
    yolo_model = YOLO("weight/best_seg.pt")

    # 加载相机参数
    params = get_all_camera_params()
    head_matrix, head_dist, head_proj = params['head']
    top_matrix, top_dist, _ = params['top']
    trans, quat = params['extrinsics']
    rot = Rotation.from_quat(quat).as_matrix()
    # 假设外参是head到top的变换，计算top到head的逆变换
    # rot_head_to_top = Rotation.from_quat(quat).as_matrix()
    # trans_head_to_top = trans
    # rot = rot_head_to_top.T  # top到head的旋转
    # trans = -rot_head_to_top.T @ trans_head_to_top  # top到head的平移

    # 设置参数
    GRID_ROWS = 4
    GRID_COLS = 6
    ASSUMED_DEPTH = 1.55

    # 启动双相机线程
    frame_queue_head = queue.Queue(maxsize=10) 
    frame_queue_top = queue.Queue(maxsize=10)
    thread_head = threading.Thread(target=capture_frames, args=(camera_head, frame_queue_head))
    thread_top = threading.Thread(target=capture_frames, args=(camera_top, frame_queue_top))
    thread_head.start()
    thread_top.start()

    while True:
        try:
            frame_head = frame_queue_head.get(timeout=1)
            left_image = frame_head[:, :640, :] # 左侧图像
            frame_top = frame_queue_top.get(timeout=1)

            # 获取YOLO结果
            results = yolo_model.track(frame_top, persist=True, iou=0.3, tracker="bytetrack.yaml")[0]
            frame_top = results.plot()

            if results.boxes is not None:
                for i, cls in enumerate(results.boxes.cls):
                    if results.names[int(cls)] == 'sorting_tray':  # 假设类名为'sorting_tray'
                        bbox = results.boxes.xyxy[i].cpu().numpy()
                        projected_points = project_bbox_to_head(bbox, ASSUMED_DEPTH, top_matrix, head_matrix, trans, rot)
                        if projected_points is not None:
                            # 在left_image上绘制投影的bbox
                            cv2.polylines(left_image, [projected_points.astype(int)], True, (0, 255, 0), 2)


            # grid_status, tray_mask_points, result = process_yolo_results(results) # 处理yolo结果

            # if tray_mask_points is not None:
            #     projected_points = project_mask_to_head(tray_mask_points, ASSUMED_DEPTH, top_matrix, head_matrix, trans, rot)
            #     if projected_points is not None:
            #         # 在head图像上绘制投影的掩码轮廓
            #         cv2.polylines(left_image, [projected_points.astype(int)], True, (0, 255, 0), 2)

            #     frame_top = visualize_grid_with_mask(frame_top, result, grid_status, GRID_ROWS, GRID_COLS)

            cv2.imshow('YOLO Detection', frame_top)
            cv2.imshow('camera_head', left_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break   
        except queue.Empty:
            continue  # 如果队列空，继续等待        
        
    # 清理
    camera_head.cleanup()
    camera_top.cleanup()
    cv2.destroyAllWindows()