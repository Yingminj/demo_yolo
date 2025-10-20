from camera_test.camera_base import Camera
from camera_test.load_cam_params import get_all_camera_params

import cv2
import numpy as np
import threading
import queue
from scipy.spatial.transform import Rotation
import apriltag


def capture_frames(camera, queue):
    for frame in camera.get_frame():
        if not queue.full():
            queue.put(frame)
        else:
            pass  # 如果队列满,丢弃旧帧


def project_apriltag_to_head_with_pose(detection, top_matrix, head_matrix, rot, trans, tag_size):
    """
    使用AprilTag的位姿估计将角点投影到head相机
    
    Args:
        detection: AprilTag检测结果
        top_matrix: top相机内参矩阵
        head_matrix: head相机内参矩阵
        rot: 旋转矩阵 (top到head)
        trans: 平移向量 (top到head)
        tag_size: AprilTag的实际尺寸(米)
    
    Returns:
        projected_corners: 投影后的四个角点坐标 (4, 2)
        tag_depth: AprilTag的深度值
    """
    # 获取AprilTag的位姿估计
    # 需要提供相机参数: [fx, fy, cx, cy]
    camera_params = [top_matrix[0, 0], top_matrix[1, 1], 
                     top_matrix[0, 2], top_matrix[1, 2]]
    
    # 估计位姿
    pose, e0, e1 = detector.detection_pose(detection, camera_params, tag_size)
    
    # pose是一个4x4的变换矩阵,表示AprilTag在相机坐标系中的位姿
    # 提取旋转矩阵和平移向量
    R_tag_to_top = pose[:3, :3]
    t_tag_to_top = pose[:3, 3]
    
    # 计算AprilTag的深度(Z坐标)
    tag_depth = t_tag_to_top[2]
    
    # 定义AprilTag在其自身坐标系中的四个角点(3D)
    # tag36h11的角点顺序: 左下, 右下, 右上, 左上
    half_size = tag_size / 2.0
    tag_corners_3d_local = np.array([
        [-half_size, -half_size, 0],  # 左下
        [ half_size, -half_size, 0],  # 右下
        [ half_size,  half_size, 0],  # 右上
        [-half_size,  half_size, 0]   # 左上
    ], dtype=np.float32)
    
    # 将AprilTag角点从标签坐标系转换到top相机坐标系
    corners_3d_top = (R_tag_to_top @ tag_corners_3d_local.T).T + t_tag_to_top
    
    # 将3D点从top相机坐标系转换到head相机坐标系
    corners_3d_head = (rot @ corners_3d_top.T).T + trans
    
    # 投影到head相机图像平面
    # 使用相机内参矩阵进行投影
    corners_2d_head = (head_matrix @ corners_3d_head.T).T
    
    # 归一化齐次坐标
    projected_corners = corners_2d_head[:, :2] / corners_2d_head[:, 2:3]
    projected_corners = projected_corners.astype(np.int32)
    
    return projected_corners, tag_depth


if __name__ == "__main__":
    # 初始化相机
    camera_head = Camera(device="/dev/video0", fps=60, width=1280, height=480, undistortion=True)
    camera_top = Camera(device="/dev/video2", fps=60)
    
    # 初始化AprilTag检测器
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    # 加载相机参数
    params = get_all_camera_params()
    head_matrix, head_dist, head_proj = params['head']
    top_matrix, top_dist, top_proj = params['top']
    trans, quat = params['extrinsics']
    rot = Rotation.from_quat(quat).as_matrix()

    # AprilTag实际尺寸(米) - 根据您的实际标签尺寸修改
    TAG_SIZE = 0.15  # 例如: 5cm的标签
    frame_count = 0
    # 启动双相机线程
    frame_queue_head = queue.Queue(maxsize=10) 
    frame_queue_top = queue.Queue(maxsize=10)
    thread_head = threading.Thread(target=capture_frames, args=(camera_head, frame_queue_head))
    thread_top = threading.Thread(target=capture_frames, args=(camera_top, frame_queue_top))
    thread_head.start()
    thread_top.start()

    print("开始AprilTag检测和投影...")
    print(f"AprilTag尺寸: {TAG_SIZE*100:.1f} cm")
    print("按 'q' 键退出")

    while True:
        try:
            frame_head = frame_queue_head.get(timeout=1)
            left_image = frame_head[:, :640, :].copy()  # 左侧图像
            frame_top = frame_queue_top.get(timeout=1)

            # 转换为灰度图进行AprilTag检测
            gray_top = cv2.cvtColor(frame_top, cv2.COLOR_BGR2GRAY)
            
            # 检测AprilTag
            detections = detector.detect(gray_top)

            # 处理每个检测到的AprilTag
            for detection in detections:
                # 获取角点坐标
                corners_top = detection.corners
                
                # 在top相机图像上绘制AprilTag
                for i in range(4):
                    pt1 = tuple(corners_top[i].astype(int))
                    pt2 = tuple(corners_top[(i + 1) % 4].astype(int))
                    cv2.line(frame_top, pt1, pt2, (0, 255, 0), 2)
                
                # 绘制中心点
                center = tuple(detection.center.astype(int))
                cv2.circle(frame_top, center, 5, (0, 0, 255), -1)
                
                # 使用位姿估计进行投影
                try:
                    projected_corners, tag_depth = project_apriltag_to_head_with_pose(
                        detection,
                        top_proj,
                        head_proj,
                        rot,
                        trans,
                        TAG_SIZE
                    )
                    
                    # 在top相机图像上显示深度信息
                    cv2.putText(frame_top, f'ID: {detection.tag_id} | D: {tag_depth:.2f}m', 
                              (center[0] - 50, center[1] - 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(frame_top, f'Score: {detection.decision_margin:.1f}', 
                              (center[0] - 50, center[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # 在head相机图像上绘制投影结果
                    if projected_corners is not None:
                        # 检查投影点是否在图像范围内
                        valid_projection = True
                        for corner in projected_corners:
                            if corner[0] < 0 or corner[0] >= 640 or corner[1] < 0 or corner[1] >= 480:
                                valid_projection = False
                                break
                        
                        if valid_projection:
                            # 绘制四边形
                            cv2.polylines(left_image, [projected_corners], 
                                        isClosed=True, color=(0, 255, 0), thickness=2)
                            
                            # 绘制角点
                            for i, corner in enumerate(projected_corners):
                                cv2.circle(left_image, tuple(corner), 4, (255, 0, 0), -1)
                                # 标记角点顺序
                                cv2.putText(left_image, str(i), 
                                          (corner[0] + 5, corner[1] + 5),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            
                            # 计算投影后的中心点
                            projected_center = np.mean(projected_corners, axis=0).astype(int)
                            cv2.circle(left_image, tuple(projected_center), 5, (0, 0, 255), -1)
                            
                            # 添加ID和深度标签
                            cv2.putText(left_image, f'ID: {detection.tag_id}', 
                                      (projected_center[0] - 40, projected_center[1] - 20),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(left_image, f'Depth: {tag_depth:.2f}m', 
                                      (projected_center[0] - 40, projected_center[1] + 5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        else:
                            print(f"Tag {detection.tag_id} projected outside image bounds")
                
                except Exception as e:
                    print(f"Error processing tag {detection.tag_id}: {e}")
                    continue

            # 显示检测到的AprilTag数量
            cv2.putText(frame_top, f'Tags: {len(detections)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(left_image, f'Projected Tags: {len(detections)}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示结果
            cv2.imshow('Top Camera - AprilTag Detection', frame_top)
            cv2.imshow('Head Camera - Projected AprilTag', left_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
               
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames") 
        except queue.Empty:
            continue  # 如果队列空,继续等待
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
    # 清理
    camera_head.cleanup()
    camera_top.cleanup()
    cv2.destroyAllWindows()