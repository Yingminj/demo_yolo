import cv2
import numpy as np

def visualize_grid_with_mask(frame, result, grid_status, grid_rows=4, grid_cols=6): # table_grid

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

def draw_3d_box(image, corners, color=(255, 255, 0), thickness=1):
    """
    在图像上绘制3D框
    
    Args:
        image: 输入图像
        corners: 8个角点坐标 (8, 2)
        color: 线条颜色
        thickness: 线条粗细
    """
    # 绘制底面
    for i in range(4):
        cv2.line(image, tuple(corners[i]), tuple(corners[(i+1)%4]), color, thickness)
    
    # 绘制顶面
    for i in range(4, 8):
        cv2.line(image, tuple(corners[i]), tuple(corners[4+(i+1)%4]), color, thickness)
    
    # 绘制竖直边
    for i in range(4):
        cv2.line(image, tuple(corners[i]), tuple(corners[i+4]), color, thickness)