import cv2
import numpy as np

def order_points(pts):
    """排序四个点：左上、右上、右下、左下"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    rect[0] = pts[np.argmin(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(s)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def fit_rotated_rect_from_mask(mask_points):
    """从mask点云拟合旋转矩形"""
    if len(mask_points) < 5:
        return None
    
    rect = cv2.minAreaRect(mask_points.astype(np.float32))
    box = cv2.boxPoints(rect)
    return np.int32(box), rect


def compute_grid_transform(mask_points, grid_rows, grid_cols):
    """计算网格变换矩阵"""
    result = fit_rotated_rect_from_mask(mask_points)
    if result is None:
        return None, None, None, None, None
    
    box, _ = result
    corner_pts = order_points(box.astype(np.float32))
    
    width = np.linalg.norm(corner_pts[0] - corner_pts[1])
    height = np.linalg.norm(corner_pts[0] - corner_pts[3])
    
    src_pts = np.array([corner_pts[0], corner_pts[1], corner_pts[3]], dtype=np.float32)
    dst_pts = np.array([[0, 0], [width, 0], [0, height]], dtype=np.float32)
    
    M = cv2.getAffineTransform(src_pts, dst_pts)
    M_inv = cv2.invertAffineTransform(M)
    
    return M, M_inv, corner_pts, width, height


def get_object_grid_position(bbox, M, width, height, grid_rows, grid_cols):
    """计算物体在哪个网格中"""
    if M is None:
        return None
    
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    center_pt = np.array([[center_x, center_y]], dtype=np.float32)
    
    transformed_pt = cv2.transform(center_pt.reshape(1, -1, 2), M)[0][0]
    rel_x, rel_y = transformed_pt
    
    grid_width = width / grid_cols
    grid_height = height / grid_rows
    
    if 0 <= rel_x < width and 0 <= rel_y < height:
        col = int(rel_x // grid_width)
        row = int(rel_y // grid_height)
        idx = row * grid_cols + col
        
        if 0 <= idx < grid_rows * grid_cols:
            return idx
    
    return None


def process_yolo_results(results, grid_rows, grid_cols, TRAY_CLASS_NAME, OBJECT_CLASS_NAMES):
    """处理YOLO检测结果，返回网格状态"""
    total_grids = grid_rows * grid_cols
    grid_status = ['free'] * total_grids
    grid_transform = (None, None, None, None, None)
    tray_mask_points = None
    
    if results.masks is not None:
        for idx, mask in enumerate(results.masks):
            cls = int(results.boxes[idx].cls.item())
            class_name = results.names[cls]
            
            if class_name == TRAY_CLASS_NAME:
                mask_points = mask.xy[0].astype(np.int32)
                grid_transform = compute_grid_transform(mask_points, grid_rows, grid_cols)
                
                if grid_transform[0] is not None:
                    tray_mask_points = mask_points
                break
    
    if results.boxes is not None and grid_transform[0] is not None:
        M, M_inv, corner_pts, width, height = grid_transform
        
        for box in results.boxes:
            cls = int(box.cls.item())
            class_name = results.names[cls]
            
            if class_name in OBJECT_CLASS_NAMES:
                bbox = box.xyxy[0].cpu().numpy()
                grid_idx = get_object_grid_position(bbox, M, width, height, grid_rows, grid_cols)
                
                if grid_idx is not None:
                    grid_status[grid_idx] = 'occ'
    
    return grid_status, tray_mask_points, grid_transform