import cv2
import numpy as np
from vis.projected import project_3d_box_to_head

def draw_3d_box(image, corners, color=(255, 255, 0), thickness=1):
    """绘制3D框到图像"""
    for i in range(4):
        cv2.line(image, tuple(corners[i]), tuple(corners[(i+1)%4]), color, thickness)
        cv2.line(image, tuple(corners[i+4]), tuple(corners[4+(i+1)%4]), color, thickness)
        cv2.line(image, tuple(corners[i]), tuple(corners[i+4]), color, thickness)


def process_objects(frame, results, top_matrix, head_matrix, rvec, tvec, assumed_depth, OBJECT_CONFIGS):
    """处理检测到的物体并绘制3D框"""
    if results.boxes is None or len(results.boxes) == 0:
        return
    
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    
    for box, cls in zip(boxes, classes):
        class_name = results.names[int(cls)]
        
        if class_name not in OBJECT_CONFIGS:
            continue
        
        config = OBJECT_CONFIGS[class_name]
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        
        corners_3d = project_3d_box_to_head(
            [center_x, center_y], config['size'], 
            top_matrix, head_matrix, rvec, tvec, assumed_depth
        )
        
        draw_3d_box(frame, corners_3d, config['color'], thickness=1)
        
        label_pos = tuple(corners_3d[4])
        cv2.putText(frame, class_name, (int(label_pos[0]), int(label_pos[1])-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config['color'], 1)
        

def draw_grids(frame, projected_grids):
    """绘制网格到图像"""
    for projected_points, status in projected_grids:
        color = (0, 0, 255) if status == 'occ' else (0, 255, 0)
        cv2.polylines(frame, [projected_points], isClosed=True, color=color, thickness=1)
        
        overlay = frame.copy()
        cv2.fillPoly(overlay, [projected_points], color)
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)