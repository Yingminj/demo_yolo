from camera_test.camera_base import Camera

import cv2
import datetime
import os
import time
from ultralytics import YOLO
import numpy as np
from scipy.spatial import ConvexHull

def fit_rotated_rect_from_mask(mask_points): # mask process
    """
    从mask点云拟合旋转矩形，具有抗遮挡能力
    """
    if len(mask_points) < 5:
        return None
    
    # 使用minAreaRect拟合最小外接旋转矩形
    rect = cv2.minAreaRect(mask_points.astype(np.float32))
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    return box, rect

def order_points(pts): # process
    """
    排序四个点：左上、右上、右下、左上
    """
    # 按x+y排序找到左上和右下
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]  # 左上
    rect[2] = pts[np.argmax(s)]  # 右下
    
    # 按y-x排序找到右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上
    rect[3] = pts[np.argmax(diff)]  # 左下
    
    return rect

def compute_grid_transform(mask_points, grid_rows=4, grid_cols=6): # process
    """
    从mask点云计算网格变换矩阵
    
    Args:
        mask_points: mask的xy坐标点
        grid_rows: 网格行数
        grid_cols: 网格列数
    
    Returns:
        M: 从图像坐标到网格坐标的仿射变换矩阵
        M_inv: 逆变换矩阵
        corner_pts: 排序后的四个角点
        width, height: 托盘宽高（像素）
    """
    result = fit_rotated_rect_from_mask(mask_points)
    if result is None:
        return None, None, None, None, None
    
    box, rect = result
    
    # 排序角点
    corner_pts = order_points(box.astype(np.float32))
    
    # 计算宽高
    width = np.linalg.norm(corner_pts[0] - corner_pts[1])
    height = np.linalg.norm(corner_pts[0] - corner_pts[3])
    
    # 源点（图像中的托盘角点）
    src_pts = np.array([
        corner_pts[0],  # 左上
        corner_pts[1],  # 右上
        corner_pts[3]   # 左下
    ], dtype=np.float32)
    
    # 目标点（标准化的矩形）
    dst_pts = np.array([
        [0, 0],
        [width, 0],
        [0, height]
    ], dtype=np.float32)
    
    # 计算仿射变换
    M = cv2.getAffineTransform(src_pts, dst_pts)
    M_inv = cv2.invertAffineTransform(M)
    
    return M, M_inv, corner_pts, width, height

def visualize_grid_with_mask(frame, result, grid_status, grid_rows=4, grid_cols=6):
    """
    基于mask可视化网格状态
    """
    # result = compute_grid_transform(mask_points, grid_rows, grid_cols)
    if result[0] is None:
        return frame
    
    M, M_inv, corner_pts, width, height = result
    
    grid_width = width / grid_cols
    grid_height = height / grid_rows
    
    # 绘制托盘轮廓
    cv2.polylines(frame, [corner_pts.astype(np.int32)], True, (255, 255, 0), 2)
    
    # 绘制网格
    for idx in range(grid_rows * grid_cols):
        row = idx // grid_cols
        col = idx % grid_cols
        
        # 网格角点（在标准化空间中）
        grid_pts = np.array([
            [col * grid_width, row * grid_height],
            [(col + 1) * grid_width, row * grid_height],
            [(col + 1) * grid_width, (row + 1) * grid_height],
            [col * grid_width, (row + 1) * grid_height]
        ], dtype=np.float32)
        
        # 变换回图像空间
        transformed_pts = cv2.transform(grid_pts.reshape(1, -1, 2), M_inv)[0]
        pts = transformed_pts.astype(np.int32)
        
        # 根据占用状态选择颜色
        color = (0, 255, 0) if grid_status[idx] == 'free' else (0, 0, 255)
        cv2.polylines(frame, [pts], True, color, 2)
        
        # 绘制网格编号
        center_grid = np.mean(pts, axis=0).astype(int)
        cv2.putText(frame, str(idx), (center_grid[0] - 10, center_grid[1] + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

def get_object_grid_position(bbox, M, width, height, grid_rows=4, grid_cols=6): # process
    """
    计算物体在哪个网格中
    """
    if M is None:
        return None
    
    # 计算bbox中心
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    center_pt = np.array([[center_x, center_y]], dtype=np.float32)
    
    # 变换到标准化空间
    transformed_pt = cv2.transform(center_pt.reshape(1, -1, 2), M)[0][0]
    rel_x, rel_y = transformed_pt
    
    grid_width = width / grid_cols
    grid_height = height / grid_rows
    
    # 检查是否在托盘范围内
    if 0 <= rel_x < width and 0 <= rel_y < height:
        col = int(rel_x // grid_width)
        row = int(rel_y // grid_height)
        idx = row * grid_cols + col
        
        if 0 <= idx < grid_rows * grid_cols:
            return idx
    
    return None

if __name__ == "__main__":
    camera = Camera(device="/dev/video2", fps=60)
    yolo_model = YOLO("weight/best_seg.pt")
    if not os.path.exists("cap"):
        os.makedirs("cap")

    GRID_ROWS = 4
    GRID_COLS = 6
    TOTAL_GRIDS = GRID_ROWS * GRID_COLS

    for frame in camera.get_frame():
        results = yolo_model.track(frame, persist=True, iou=0.3, tracker="bytetrack.yaml")[0]
        
        grid_status = ['free'] * TOTAL_GRIDS
        
        tray_M = None
        tray_M_inv = None
        tray_width = None
        tray_height = None
        tray_mask_points = None
        
        # 首先找到sorting_tray并建立网格
        if results.masks is not None:
            for idx, mask in enumerate(results.masks):
                cls = int(results.boxes[idx].cls.item())
                class_name = results.names[cls]
                
                if class_name == 'sorting_tray':
                    # 获取mask的xy坐标
                    mask_points = mask.xy[0].astype(np.int32)
                    
                    # 计算网格变换
                    result = compute_grid_transform(mask_points, GRID_ROWS, GRID_COLS)
                    if result[0] is not None:
                        tray_M, tray_M_inv, corner_pts, tray_width, tray_height = result
                        tray_mask_points = mask_points
                    break
        
        # 然后检测物体并分配到网格
        if results.boxes is not None and tray_M is not None:
            for box in results.boxes:
                cls = int(box.cls.item())
                class_name = results.names[cls]
                
                if class_name in ['cola', 'mug']:
                    bbox = box.xyxy[0].cpu().numpy()
                    grid_idx = get_object_grid_position(
                        bbox, tray_M, tray_width, tray_height, 
                        GRID_ROWS, GRID_COLS
                    )
                    
                    if grid_idx is not None:
                        grid_status[grid_idx] = 'occ'
        
        # 可视化网格
        if tray_mask_points is not None:
            frame = visualize_grid_with_mask(
                frame, result, grid_status, 
                GRID_ROWS, GRID_COLS
            )
        
        cv2.imshow('YOLO Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.cleanup()