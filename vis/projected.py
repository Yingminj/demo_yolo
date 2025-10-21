import cv2
import numpy as np

def project_grid_to_head(grid_transform, grid_status, grid_rows, grid_cols, 
                         top_matrix, head_matrix, rvec, tvec, assumed_depth):
    """投影网格到head相机"""
    if grid_transform[0] is None:
        return []
    
    M, M_inv, _, width, height = grid_transform
    grid_width = width / grid_cols
    grid_height = height / grid_rows
    projected_grids = []
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            idx = row * grid_cols + col
            status = grid_status[idx]
            
            x1, y1 = col * grid_width, row * grid_height
            x2, y2 = (col + 1) * grid_width, (row + 1) * grid_height
            
            grid_corners_norm = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            grid_corners_top = cv2.transform(grid_corners_norm, M_inv).reshape(-1, 2)
            corners_2d = grid_corners_top.reshape(-1, 1, 2).astype(np.float32)
            normalized_points = cv2.undistortPoints(corners_2d, top_matrix, None)
            
            corners_3d = np.zeros((4, 3), dtype=np.float32)
            corners_3d[:, :2] = normalized_points.reshape(-1, 2) * assumed_depth
            corners_3d[:, 2] = assumed_depth
            
            projected_points, _ = cv2.projectPoints(corners_3d, rvec, tvec, head_matrix, None)
            projected_points = projected_points.reshape(-1, 2).astype(np.int32)
            
            projected_grids.append((projected_points, status))
    
    return projected_grids

def project_3d_box_to_head(center_2d, box_size, top_matrix, head_matrix, rvec, tvec, assumed_depth):
    """投影3D框到head相机"""
    L, W, H = box_size
    
    center_pt = np.array([[center_2d]], dtype=np.float32)
    normalized_center = cv2.undistortPoints(center_pt, top_matrix, None)
    
    center_3d = np.zeros(3, dtype=np.float32)
    center_3d[:2] = normalized_center.reshape(2) * assumed_depth
    center_3d[2] = assumed_depth
    
    corners_3d = np.array([
        [-L/2, -W/2, 0], [L/2, -W/2, 0], [L/2, W/2, 0], [-L/2, W/2, 0],
        [-L/2, -W/2, -H], [L/2, -W/2, -H], [L/2, W/2, -H], [-L/2, W/2, -H],
    ], dtype=np.float32)
    
    corners_3d += center_3d
    projected_points, _ = cv2.projectPoints(corners_3d, rvec, tvec, head_matrix, None)
    
    return projected_points.reshape(-1, 2).astype(np.int32)