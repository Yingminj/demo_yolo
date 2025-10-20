from camera_test.camera_base import Camera
from camera_test.load_cam_params import get_all_camera_params

import cv2
import datetime
import os
import time
from ultralytics import YOLO
import numpy as np
from scipy.spatial import ConvexHull
import rclpy.time
from scene_process.table_grid import process_yolo_results
from vis.vis import visualize_grid_with_mask, draw_3d_box
from vis.projected import project_grid_to_head_optimized, project_3d_box_to_head_optimized
import threading
import queue
from scipy.spatial.transform import Rotation

# ROS2相关导入
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs


## 1发布图像和相机信息，2获取左右手在head_left_camera坐标系下的位姿
class ImagePublisher(Node):
    def __init__(self, head_matrix, head_dist, head_proj):
        super().__init__('head_camera_publisher')
        
        # 创建发布者
        self.image_pub = self.create_publisher(Image, '/head_camera/image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/head_camera/camera_info', 10)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # 相机参数
        self.head_matrix = head_matrix
        self.head_dist = head_dist
        self.head_proj = head_proj
        
        # TF2相关
        self.tf_buffer = Buffer()
        # self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=0.1))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 存储最新的左右手位姿
        self.left_hand_pose = None
        self.right_hand_pose = None
        
    def get_hand_poses(self):
        """获取左右手在head_left_camera坐标系下的位姿"""
        try:
            now = self.get_clock().now()
            timeout = rclpy.duration.Duration(seconds=0.1)

            # 获取left_link相对于head_left_camera的变换
            left_transform = self.tf_buffer.lookup_transform(
                'head_left_camera',
                'left_tool',
                rclpy.time.Time()
            )
            
            # 获取right_link相对于head_left_camera的变换
            right_transform = self.tf_buffer.lookup_transform(
                'head_left_camera',
                'right_tool',
                rclpy.time.Time()
            )
            
            # 转换为位姿（translation + rotation）
            self.left_hand_pose = self._transform_to_pose(left_transform)
            self.right_hand_pose = self._transform_to_pose(right_transform)
            
            return True
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'Failed to get hand poses: {str(e)}')
            return False
    
    def _transform_to_pose(self, transform):
        """将TF变换转换为位姿(position + rotation matrix)"""
        # 提取平移
        translation = np.array([
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z
        ])
        # translation[1]-=0.1
        
        # 提取旋转（四元数转旋转矩阵）
        quat = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]

        # 四元数 -> 欧拉角 (roll, pitch, yaw)
        r = Rotation.from_quat(quat)
        euler = r.as_euler('xyz', degrees=False)  # 返回 [roll, pitch, yaw] 弧度制
        # euler[1]+=0.2
        
        # 欧拉角 -> 四元数
        r_new = Rotation.from_euler('xyz', euler, degrees=False)
        quat_new = r_new.as_quat()  # 返回 [x, y, z, w]
        
        rotation = Rotation.from_quat(quat_new).as_matrix()
        
        return {'translation': translation, 'rotation': rotation}
        
    def publish_image(self, left_image):
        """发布图像和相机信息"""
        try:
            # 转换并发布图像
            ros_image = self.bridge.cv2_to_imgmsg(left_image, encoding='bgr8')
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = 'head_left_camera'
            self.image_pub.publish(ros_image)
            
            # 发布相机信息
            camera_info = CameraInfo()
            camera_info.header = ros_image.header
            camera_info.height = left_image.shape[0]
            camera_info.width = left_image.shape[1]
            
            # 内参矩阵 K (3x3) -> 9个元素
            camera_info.k = self.head_matrix.flatten().tolist()
            
            # 畸变系数 D
            camera_info.d = self.head_dist.flatten().tolist()
            camera_info.distortion_model = 'plumb_bob'
            
            # 投影矩阵 P (3x4) -> 12个元素
            if self.head_proj.shape == (3, 4):
                camera_info.p = self.head_proj.flatten().tolist()
            elif self.head_proj.shape == (3, 3):
                proj_matrix = np.hstack([self.head_proj, np.zeros((3, 1))])
                camera_info.p = proj_matrix.flatten().tolist()
            else:
                proj_matrix = np.hstack([self.head_matrix, np.zeros((3, 1))])
                camera_info.p = proj_matrix.flatten().tolist()
            
            camera_info.p = [float(x) for x in camera_info.p[:12]]
            
            # 单位矩阵作为R (3x3) -> 9个元素
            camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            
            self.camera_info_pub.publish(camera_info)
            
        except Exception as e:
            self.get_logger().error(f'Failed to publish image: {str(e)}')

def create_hand_cube_corners(pose, size=0.05):
    """创建手部立方体的8个角点（在相机坐标系下）
    
    Args:
        pose: 包含translation和rotation的字典
        size: 立方体边长（米）
    
    Returns:
        corners_3d: 8x3的numpy数组，表示立方体的8个角点
    """
    # 定义立方体的8个角点（以手部中心为原点）
    half_size = size / 2
    # local_corners = np.array([
    #     [-half_size, -half_size, -half_size],
    #     [half_size, -half_size, -half_size],
    #     [half_size, half_size, -half_size],
    #     [-half_size, half_size, -half_size],
    #     [-half_size, -half_size, half_size],
    #     [half_size, -half_size, half_size],
    #     [half_size, half_size, half_size],
    #     [-half_size, half_size, half_size]
    # ])
    L,W,H = 0.20,0.10,0.10
    local_corners = np.array([
        [-L/2, -W/2, 0],
        [L/2, -W/2, 0],
        [L/2, W/2, 0],
        [-L/2, W/2, 0],
        [-L/2, -W/2, -H],
        [L/2, -W/2, -H],
        [L/2, W/2, -H],
        [-L/2, W/2, -H],
    ], dtype=np.float32)

    # 应用旋转和平移，转换到相机坐标系
    corners_3d = (pose['rotation'] @ local_corners.T).T + pose['translation']
    
    return corners_3d

def project_hand_to_image(corners_3d, camera_matrix):
    """将3D角点投影到图像平面
    
    Args:
        corners_3d: 8x3的numpy数组
        camera_matrix: 3x3的相机内参矩阵
    
    Returns:
        corners_2d: 8x2的numpy数组，图像坐标
    """
    # 投影到图像平面
    corners_2d = camera_matrix @ corners_3d.T
    corners_2d = corners_2d[:2, :] / corners_2d[2, :]
    corners_2d = corners_2d.T.astype(np.int32)
    
    return corners_2d

def draw_hand_cube(image, corners_2d, color=(255, 0, 0), thickness=2):
    """在图像上绘制手部立方体
    
    Args:
        image: 输入图像
        corners_2d: 8x2的numpy数组
        color: BGR颜色
        thickness: 线条粗细
    """
    # 定义立方体的12条边
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 竖边
    ]
    
    # 绘制每条边
    for start, end in edges:
        pt1 = tuple(corners_2d[start])
        pt2 = tuple(corners_2d[end])
        cv2.line(image, pt1, pt2, color, thickness)

def capture_frames(camera, queue):
    for frame in camera.get_frame():
        if not queue.full():
            queue.put(frame)
        else:
            pass

if __name__ == "__main__":
    # 初始化ROS2
    rclpy.init()
    
    # 初始化相机和模型
    camera_head = Camera(device="/dev/video2", fps=30, width=1280, height=480, undistortion=True)
    camera_top = Camera(device="/dev/video0", fps=30)
    yolo_model = YOLO("weight/best_seg.pt")

    # 加载相机参数
    params = get_all_camera_params()
    head_matrix, head_dist, head_proj = params['head']
    top_matrix, top_dist, top_proj = params['top']
    trans, quat = params['extrinsics']
    rot = Rotation.from_quat(quat).as_matrix()
    
    # 创建ROS2图像发布器
    image_publisher = ImagePublisher(head_matrix, head_dist, head_proj)
    
    # 预计算旋转向量
    rvec_global, _ = cv2.Rodrigues(rot)
    tvec_global = trans.reshape(3, 1)

    # 设置参数
    GRID_ROWS = 4
    GRID_COLS = 6
    ASSUMED_DEPTH = 1.48
    HAND_CUBE_SIZE = 0.1  # 手部立方体边长（米）

    # 启动双相机线程
    frame_queue_head = queue.Queue(maxsize=10) 
    frame_queue_top = queue.Queue(maxsize=10)
    thread_head = threading.Thread(target=capture_frames, args=(camera_head, frame_queue_head), daemon=True)
    thread_top = threading.Thread(target=capture_frames, args=(camera_top, frame_queue_top), daemon=True)
    thread_head.start()
    thread_top.start()

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
                results = yolo_model.track(frame_top, persist=True, iou=0.3, tracker="bytetrack.yaml", verbose=False)[0]

                # 处理网格和物体检测
                grid_status, tray_mask_points, grid_transform = process_yolo_results(results)
                
                # 投影网格到head相机
                projected_grids = project_grid_to_head_optimized(
                    grid_transform, grid_status, GRID_ROWS, GRID_COLS,
                    top_proj, head_proj, rvec_global, tvec_global, ASSUMED_DEPTH
                )
                
                # 绘制投影的网格
                for projected_points, status in projected_grids:
                    if status == 'occ':
                        color = (0, 0, 255)
                    else:
                        color = (0, 255, 0)
                    
                    cv2.polylines(left_image, [projected_points], isClosed=True, color=color, thickness=1)
                    
                    overlay = left_image.copy()
                    cv2.fillPoly(overlay, [projected_points], color)
                    cv2.addWeighted(overlay, 0.2, left_image, 0.8, 0, left_image)
                    del overlay

                # 处理检测到的物体
                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy()
                    names = results.names
                    
                    for box, cls in zip(boxes, classes):
                        class_name = names[int(cls)]
                        
                        center_x = (box[0] + box[2]) / 2
                        center_y = (box[1] + box[3]) / 2
                        
                        if class_name == 'cola':
                            box_size = [0.06, 0.06, 0.16]
                            color = (0, 255, 255)
                        elif class_name == 'mug':
                            box_size = [0.08, 0.08, 0.07]
                            color = (255, 0, 255)
                        else:
                            continue
                        
                        corners_3d = project_3d_box_to_head_optimized(
                            [center_x, center_y], box_size,
                            top_proj, head_proj, rvec_global, tvec_global, ASSUMED_DEPTH
                        )
                        
                        draw_3d_box(left_image, corners_3d, color, thickness=1)
                        
                        label = f'{class_name}'
                        label_pos = tuple(corners_3d[4])
                        cv2.putText(left_image, label, (int(label_pos[0]), int(label_pos[1]-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 获取并绘制左右手
                # if image_publisher.get_hand_poses():
                #     # 绘制左手（蓝色）
                #     if image_publisher.left_hand_pose is not None:
                #         left_corners_3d = create_hand_cube_corners(image_publisher.left_hand_pose, HAND_CUBE_SIZE)
                #         left_corners_2d = project_hand_to_image(left_corners_3d, head_matrix)
                #         draw_hand_cube(left_image, left_corners_2d, color=(255, 0, 0), thickness=2)
                        
                #         # 添加标签
                #         center_2d = np.mean(left_corners_2d, axis=0).astype(np.int32)
                #         cv2.putText(left_image, 'Left Hand', tuple(center_2d), 
                #                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
                #     # 绘制右手（绿色）
                #     if image_publisher.right_hand_pose is not None:
                #         right_corners_3d = create_hand_cube_corners(image_publisher.right_hand_pose, HAND_CUBE_SIZE)
                #         right_corners_2d = project_hand_to_image(right_corners_3d, head_matrix)
                #         draw_hand_cube(left_image, right_corners_2d, color=(0, 255, 0), thickness=2)
                        
                #         # 添加标签
                #         center_2d = np.mean(right_corners_2d, axis=0).astype(np.int32)
                #         cv2.putText(left_image, 'Right Hand', tuple(center_2d), 
                #                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 发布图像到ROS2
                # image_publisher.publish_image(left_image)

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