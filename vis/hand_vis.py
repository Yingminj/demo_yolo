import cv2
import numpy as np

def create_hand_cube_corners(pose, size=0.05):
    """创建手部立方体的8个角点（在相机坐标系下）"""

    L, W, H = 0.20, 0.10, 0.10
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
    """将3D角点投影到图像平面"""

    # 投影到图像平面
    corners_2d = camera_matrix @ corners_3d.T
    corners_2d = corners_2d[:2, :] / corners_2d[2, :]
    corners_2d = corners_2d.T.astype(np.int32)
    
    return corners_2d


def draw_hand_cube(image, corners_2d, color=(255, 0, 0), thickness=2):
    """在图像上绘制手部立方体"""

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
