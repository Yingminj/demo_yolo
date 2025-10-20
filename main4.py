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
from vis.vis import visualize_grid_with_mask, draw_3d_box
from vis.projected import project_grid_to_head_optimized, project_3d_box_to_head_optimized
import threading
import queue
from scipy.spatial.transform import Rotation

def capture_frames(camera, queue):
    for frame in camera.get_frame():
        if not queue.full():
            queue.put(frame)
        else:
            pass # 如果队列满,丢弃旧帧或处理

if __name__ == "__main__":
    # 初始化相机和模型
    camera_head = Camera(device="/dev/video2", fps=30, width=1280, height= 480, undistortion=True)
    camera_top = Camera(device="/dev/video0", fps=30)
    yolo_model = YOLO("weight/best_seg.pt")

    # 加载相机参数
    params = get_all_camera_params()
    head_matrix, head_dist, head_proj = params['head']
    top_matrix, top_dist, top_proj = params['top']
    # trans, quat = params['extrinsics']
    # rot = Rotation.from_quat(quat).as_matrix()
    # rvec_global, _ = cv2.Rodrigues(rot)
    # tvec_global = trans.reshape(3, 1)
    trans, quat = params['extrinsics']
    rvec_global = Rotation.from_quat(quat).as_rotvec().reshape(3, 1)
    tvec_global = trans.reshape(3, 1)
    
    # 设置参数
    GRID_ROWS = 4
    GRID_COLS = 6
    ASSUMED_DEPTH = 1.50

    # 启动双相机线程
    frame_queue_head = queue.Queue(maxsize=10) 
    frame_queue_top = queue.Queue(maxsize=10)
    thread_head = threading.Thread(target=capture_frames, args=(camera_head, frame_queue_head), daemon=True)
    thread_top = threading.Thread(target=capture_frames, args=(camera_top, frame_queue_top), daemon=True)
    thread_head.start()
    thread_top.start()

    # 添加帧计数器用于调试
    frame_count = 0
    
    try:
        while True:
            try:

                frame_head = frame_queue_head.get(timeout=1)
                left_image = frame_head[:, :640, :].copy()
                frame_top = frame_queue_top.get(timeout=1)

                # 获取YOLO结果
                results = yolo_model.track(frame_top, persist=True, iou=0.3, tracker="bytetrack.yaml",verbose=False)[0]
                frame_top = results.plot()

                # # 处理网格和物体检测
                grid_status, tray_mask_points, grid_transform = process_yolo_results(results)
                # 如果未检测到托盘，跳过投影与绘制网格
                if grid_transform is None:
                    cv2.imshow('YOLO Detection', frame_top)
                    cv2.imshow('camera_head', left_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    frame_count += 1
                    continue

                # 投影网格到head相机(优化版)
                projected_grids = project_grid_to_head_optimized(grid_transform, grid_status, GRID_ROWS, GRID_COLS,top_proj, head_proj, rvec_global, tvec_global, ASSUMED_DEPTH)
                
                # 绘制投影的网格
                for projected_points, status in projected_grids:
                    # 根据状态选择颜色
                    if status == 'occ':
                        color = (0, 0, 255)  # 红色表示被占用
                    else:
                        color = (0, 255, 0)  # 绿色表示空闲
                    
                    # 绘制网格边界
                    cv2.polylines(left_image, [projected_points], isClosed=True, color=color, thickness=1)
                    
                    # 可选: 填充半透明区域(优化内存使用)
                    overlay = left_image.copy()
                    cv2.fillPoly(overlay, [projected_points], color)
                    cv2.addWeighted(overlay, 0.2, left_image, 0.8, 0, left_image)
                    del overlay  # 显式释放

                # 处理检测到的物体，绘制3D框
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy()
                    names = results.names
                    
                    for box, cls in zip(boxes, classes):
                        class_name = names[int(cls)]
                        
                        # 计算bbox中心
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
                        
                        # 根据类别设置3D框尺寸
                        if class_name == 'cola':
                            box_size = [0.06, 0.06, 0.16]  # L*W*H (米)
                            color = (0, 255, 255)  # 黄色
                        elif class_name == 'mug':
                            box_size = [0.08, 0.08, 0.07]  # L*W*H (米)
                            color = (255, 0, 255)  # 紫色
                        else:
                            continue
                        
                        # 投影3D框到head相机(使用预计算的rvec和tvec)
                        corners_3d = project_3d_box_to_head_optimized(
                            [center_x, center_y], box_size,
                            top_proj, head_proj, rvec_global, tvec_global, ASSUMED_DEPTH
                        )
                        
                        # 绘制3D框
                        draw_3d_box(left_image, corners_3d, color, thickness=1)
                        
                        # 添加标签
                        label = f'{class_name}'
                        label_pos = tuple(corners_3d[4])
                        cv2.putText(left_image, label, (int(label_pos[0]), int(label_pos[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 显示图像
                cv2.imshow('YOLO Detection', frame_top)
                cv2.imshow('camera_head', left_image)

                # 增加waitKey时间以确保UI响应
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")
                
            except queue.Empty:
                print("Queue empty, waiting...")
                continue
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # 清理
        print("Cleaning up...")
        camera_head.cleanup()
        camera_top.cleanup()
        cv2.destroyAllWindows()