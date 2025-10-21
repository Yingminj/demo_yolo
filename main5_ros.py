from camera_test.camera_base import Camera
from camera_test.load_cam_params import get_all_camera_params
from ros_publisher.image_pub import ImagePublisher
from scene_process.table_grid import process_yolo_results
from vis.vis import process_objects, draw_grids
from vis.projected import project_grid_to_head
from vis.hand_vis import create_hand_cube_corners, project_hand_to_image, draw_hand_cube

import cv2
import threading
import queue
import numpy as np

from ultralytics import YOLO
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import rclpy

## 更新为ROS2版本，集成手部可视化功能 ##更新函数调用和主循环##

def capture_frames(camera, frame_queue):
    """捕获相机帧到队列"""
    for frame in camera.get_frame():
        if not frame_queue.full():
            frame_queue.put(frame)

def main():
    # 初始化ROS2
    rclpy.init()
    
    # 初始化相机和模型
    camera_head = Camera(device="/dev/video2", fps=30, width=1280, height=480, undistortion=True)
    camera_top = Camera(device="/dev/video0", fps=30)
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # 加载相机参数
    params = get_all_camera_params()
    head_matrix, head_dist, head_proj = params['head']
    top_matrix, top_dist, top_proj = params['top']
    trans, quat = params['extrinsics']
    
    # 创建ROS2图像发布器
    image_publisher = ImagePublisher(head_matrix, head_dist, head_proj)
    
    # 预计算旋转向量
    rvec = Rotation.from_quat(quat).as_rotvec().reshape(3, 1)
    tvec = trans.reshape(3, 1)

    # 启动双相机线程
    frame_queue_head = queue.Queue(maxsize=10) 
    frame_queue_top = queue.Queue(maxsize=10)
    
    threading.Thread(target=capture_frames, args=(camera_head, frame_queue_head), daemon=True).start()
    threading.Thread(target=capture_frames, args=(camera_top, frame_queue_top), daemon=True).start()

    frame_count = 0
    
    try:
        while True:
            try:
                # 处理ROS2回调
                rclpy.spin_once(image_publisher, timeout_sec=0.01)

                frame_head = frame_queue_head.get(timeout=1)
                left_image = frame_head[:, :640, :].copy()
                frame_top = frame_queue_top.get(timeout=1)

                # 获取YOLO结果
                results = yolo_model.track(frame_top, persist=True, iou=0.3, 
                                          tracker="bytetrack.yaml", verbose=False)[0]

                # 处理网格和物体检测
                grid_status, _, grid_transform = process_yolo_results(
                    results, GRID_ROWS, GRID_COLS, TRAY_CLASS_NAME, OBJECT_CLASS_NAMES
                )
                
                # 投影网格和物体到head相机
                if grid_transform[0] is not None:
                    projected_grids = project_grid_to_head(
                        grid_transform, grid_status, GRID_ROWS, GRID_COLS,
                        top_proj, head_proj, rvec, tvec, ASSUMED_DEPTH
                    )
                    draw_grids(left_image, projected_grids)
                    process_objects(left_image, results, top_proj, head_proj, 
                                  rvec, tvec, ASSUMED_DEPTH, OBJECT_CONFIGS)

                # 获取并绘制左右手
                if image_publisher.get_hand_poses():
                    # 绘制左手（蓝色）
                    if image_publisher.left_hand_pose is not None:
                        left_corners_3d = create_hand_cube_corners(
                            image_publisher.left_hand_pose, HAND_CUBE_SIZE
                        )
                        left_corners_2d = project_hand_to_image(left_corners_3d, head_matrix)
                        draw_hand_cube(left_image, left_corners_2d, color=(255, 0, 0), thickness=2)
                        
                        # 添加标签
                        center_2d = np.mean(left_corners_2d, axis=0).astype(np.int32)
                        cv2.putText(left_image, 'Left Hand', tuple(center_2d), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                    # 绘制右手（绿色）
                    if image_publisher.right_hand_pose is not None:
                        right_corners_3d = create_hand_cube_corners(
                            image_publisher.right_hand_pose, HAND_CUBE_SIZE
                        )
                        right_corners_2d = project_hand_to_image(right_corners_3d, head_matrix)
                        draw_hand_cube(left_image, right_corners_2d, color=(0, 255, 0), thickness=2)
                        
                        # 添加标签
                        center_2d = np.mean(right_corners_2d, axis=0).astype(np.int32)
                        cv2.putText(left_image, 'Right Hand', tuple(center_2d), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 发布图像到ROS2
                image_publisher.publish_image(left_image)

                # 显示图像
                cv2.imshow('camera_head', left_image)

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
        print("Cleaning up...")
        camera_head.cleanup()
        camera_top.cleanup()
        cv2.destroyAllWindows()
        image_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # 配置常量
    TRAY_CLASS_NAME = 'dish_drying_rack'
    OBJECT_CLASS_NAMES = ['cola', 'mug', 'dish']
    OBJECT_CONFIGS = {
        'cola': {'size': [0.06, 0.06, 0.16], 'color': (0, 255, 255)},
        'mug': {'size': [0.08, 0.08, 0.07], 'color': (255, 0, 255)},
        # 'dish': {'size': [0.20, 0.15, 0.02], 'color': (255, 255, 0)},
    }
    GRID_ROWS = 1
    GRID_COLS = 7
    ASSUMED_DEPTH = 1.50
    HAND_CUBE_SIZE = 0.1  # 手部立方体边长（米）

    YOLO_MODEL_PATH = "weight/best_dish.pt"
    
    # Run
    main()