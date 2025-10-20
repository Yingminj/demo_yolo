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
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from scipy.spatial.transform import Rotation

# 多进程摄像头捕获程序测试，每个摄像头一个进程，主进程处理和显示

def capture_frames_process(device, fps, frame_queue, stop_event, width=None, height=None, undistortion=False):
    """
    摄像头捕获进程
    
    Args:
        device: 摄像头设备路径
        fps: 帧率
        frame_queue: 多进程队列
        stop_event: 停止事件
        width: 图像宽度
        height: 图像高度
        undistortion: 是否去畸变
    """
    # 在子进程中初始化Camera对象
    if width and height:
        camera = Camera(device=device, fps=fps, width=width, height=height, undistortion=undistortion)
    else:
        camera = Camera(device=device, fps=fps, undistortion=undistortion)
    
    print(f"[进程-{device}] 摄像头已启动")
    frame_count = 0
    
    try:
        for frame in camera.get_frame():
            if stop_event.is_set():
                break
            
            # 添加时间戳
            timestamp = time.time()
            frame_count += 1
            
            try:
                # 如果队列满，丢弃最旧的帧
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                
                # 打包帧数据
                frame_data = {
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_id': frame_count,
                    'device': device
                }
                
                frame_queue.put(frame_data, block=False)
                
            except Exception as e:
                if not stop_event.is_set():
                    print(f"[进程-{device}] 队列放入失败: {e}")
            
            # 控制帧率
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print(f"[进程-{device}] 收到中断信号")
    except Exception as e:
        print(f"[进程-{device}] 异常: {e}")
    finally:
        camera.cleanup()
        print(f"[进程-{device}] 摄像头已释放，共捕获 {frame_count} 帧")


def process_and_display(frame_queue_head, frame_queue_top, stop_event, params_dict):
    """
    处理和显示进程（主进程）
    
    Args:
        frame_queue_head: head摄像头帧队列
        frame_queue_top: top摄像头帧队列
        stop_event: 停止事件
        params_dict: 相机参数字典
    """
    print("[主进程] 初始化YOLO模型...")
    yolo_model = YOLO("weight/best_seg.pt")
    
    # 解包参数
    head_matrix, head_dist, head_proj = params_dict['head']
    top_matrix, top_dist, top_proj = params_dict['top']
    trans, quat = params_dict['extrinsics']
    rvec_global = Rotation.from_quat(quat).as_rotvec().reshape(3, 1)
    tvec_global = trans.reshape(3, 1)
    
    # 设置参数
    GRID_ROWS = 4
    GRID_COLS = 6
    ASSUMED_DEPTH = 1.50
    
    # 帧计数器
    frame_count = 0
    last_fps_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    print("[主进程] 开始处理...")
    
    try:
        while not stop_event.is_set():
            try:
                # 从队列获取帧（带超时）
                frame_data_head = frame_queue_head.get(timeout=1)
                frame_data_top = frame_queue_top.get(timeout=1)
                
                frame_head = frame_data_head['frame']
                frame_top = frame_data_top['frame']
                
                # 计算时间差（用于调试同步问题）
                time_diff = abs(frame_data_head['timestamp'] - frame_data_top['timestamp'])
                
                # 提取左图
                left_image = frame_head[:, :640, :].copy()
                
                # 获取YOLO结果
                results = yolo_model.track(
                    frame_top, 
                    persist=True, 
                    iou=0.3, 
                    tracker="bytetrack.yaml",
                    verbose=False
                )[0]
                frame_top = results.plot()
                
                # 处理网格和物体检测
                grid_status, tray_mask_points, grid_transform = process_yolo_results(results)
                
                # 如果检测到托盘
                if grid_transform is not None:
                    # 投影网格到head相机
                    projected_grids = project_grid_to_head_optimized(
                        grid_transform, grid_status, GRID_ROWS, GRID_COLS,
                        top_proj, head_proj, rvec_global, tvec_global, ASSUMED_DEPTH
                    )
                    
                    # 绘制投影的网格
                    for projected_points, status in projected_grids:
                        color = (0, 0, 255) if status == 'occ' else (0, 255, 0)
                        cv2.polylines(left_image, [projected_points], isClosed=True, color=color, thickness=1)
                        
                        # 填充半透明区域
                        overlay = left_image.copy()
                        cv2.fillPoly(overlay, [projected_points], color)
                        cv2.addWeighted(overlay, 0.2, left_image, 0.8, 0, left_image)
                        del overlay
                    
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
                                box_size = [0.06, 0.06, 0.16]
                                color = (0, 255, 255)
                            elif class_name == 'mug':
                                box_size = [0.08, 0.08, 0.07]
                                color = (255, 0, 255)
                            else:
                                continue
                            
                            # 投影3D框
                            corners_3d = project_3d_box_to_head_optimized(
                                [center_x, center_y], box_size,
                                top_proj, head_proj, rvec_global, tvec_global, ASSUMED_DEPTH
                            )
                            
                            # 绘制3D框
                            draw_3d_box(left_image, corners_3d, color, thickness=1)
                            
                            # 添加标签
                            label = f'{class_name}'
                            label_pos = tuple(corners_3d[4])
                            cv2.putText(
                                left_image, label, 
                                (int(label_pos[0]), int(label_pos[1]-10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                            )
                
                # 添加FPS和时间差信息
                fps_counter += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    current_fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                
                # 在top图像上显示信息
                info_text = [
                    f"FPS: {current_fps:.1f}",
                    f"Frame: {frame_count}",
                    f"Time Diff: {time_diff*1000:.2f}ms"
                ]
                for i, text in enumerate(info_text):
                    cv2.putText(
                        frame_top, text, (10, 30 + i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                
                # 显示图像
                cv2.imshow('YOLO Detection', frame_top)
                cv2.imshow('camera_head', left_image)
                
                # 检查退出键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[主进程] 用户请求退出")
                    stop_event.set()
                    break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"[主进程] Processed {frame_count} frames, FPS: {current_fps:.1f}")
                
            except mp.queues.Empty:
                print("[主进程] Queue empty, waiting...")
                continue
            except Exception as e:
                print(f"[主进程] 处理异常: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    except KeyboardInterrupt:
        print("\n[主进程] 收到键盘中断")
    finally:
        cv2.destroyAllWindows()
        print("[主进程] 已退出")


def main():
    """主函数"""
    # 设置多进程启动方式
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass  # 已经设置过
    
    print("="*60)
    print("启动多进程双摄像头捕获系统")
    print("="*60)
    
    # 加载相机参数（在主进程中）
    print("加载相机参数...")
    params = get_all_camera_params()
    
    # 创建多进程队列
    frame_queue_head = Queue(maxsize=10)
    frame_queue_top = Queue(maxsize=10)
    
    # 创建停止事件
    stop_event = Event()
    
    # 创建捕获进程
    process_head = Process(
        target=capture_frames_process,
        args=(
            "/dev/video2",  # device
            30,             # fps
            frame_queue_head,
            stop_event,
            1280,           # width
            480,            # height
            True            # undistortion
        ),
        name="CaptureHead"
    )
    
    process_top = Process(
        target=capture_frames_process,
        args=(
            "/dev/video0",
            30,
            frame_queue_top,
            stop_event
        ),
        name="CaptureTop"
    )
    
    # 启动捕获进程
    print("启动摄像头捕获进程...")
    process_head.start()
    process_top.start()
    
    # 等待摄像头初始化
    time.sleep(2)
    print("摄像头初始化完成")
    
    try:
        # 在主进程中运行处理和显示（因为OpenCV窗口需要在主进程）
        process_and_display(frame_queue_head, frame_queue_top, stop_event, params)
        
    except KeyboardInterrupt:
        print("\n收到键盘中断信号，正在关闭...")
    finally:
        # 设置停止事件
        stop_event.set()
        
        # 清空队列，避免进程阻塞
        print("清空队列...")
        try:
            while not frame_queue_head.empty():
                frame_queue_head.get_nowait()
        except:
            pass
        try:
            while not frame_queue_top.empty():
                frame_queue_top.get_nowait()
        except:
            pass
        
        # 等待进程结束
        print("等待进程结束...")
        process_head.join(timeout=5)
        process_top.join(timeout=5)
        
        # 强制终止未结束的进程
        if process_head.is_alive():
            print("强制终止 Head 捕获进程")
            process_head.terminate()
            process_head.join(timeout=2)
        if process_top.is_alive():
            print("强制终止 Top 捕获进程")
            process_top.terminate()
            process_top.join(timeout=2)
        
        print("="*60)
        print("所有进程已关闭")
        print("="*60)


if __name__ == "__main__":
    main()