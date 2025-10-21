import cv2
import numpy as np
import threading
import queue
from ultralytics import YOLO
from scipy.spatial.transform import Rotation

from camera_test.camera_base import Camera
from camera_test.load_cam_params import get_all_camera_params
from scene_process.table_grid import process_yolo_results
from vis.vis import process_objects, draw_grids
from vis.projected import project_grid_to_head

def capture_frames(camera, frame_queue):
    """捕获相机帧到队列"""
    for frame in camera.get_frame():
        if not frame_queue.full():
            frame_queue.put(frame)

def main():
    
    camera_head = Camera(device="/dev/video2", fps=30, width=1280, height=480, undistortion=True)
    camera_top = Camera(device="/dev/video0", fps=30)
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    params = get_all_camera_params()
    head_matrix, _, head_proj = params['head']
    top_matrix, _, top_proj = params['top']
    trans, quat = params['extrinsics']
    rvec = Rotation.from_quat(quat).as_rotvec().reshape(3, 1)
    tvec = trans.reshape(3, 1)
    
    frame_queue_head = queue.Queue(maxsize=10)
    frame_queue_top = queue.Queue(maxsize=10)
    
    threading.Thread(target=capture_frames, args=(camera_head, frame_queue_head), daemon=True).start()
    threading.Thread(target=capture_frames, args=(camera_top, frame_queue_top), daemon=True).start()
    
    frame_count = 0
    
    try:
        while True:
            try:
                frame_head = frame_queue_head.get(timeout=1)
                left_image = frame_head[:, :640, :].copy()
                frame_top = frame_queue_top.get(timeout=1)
                
                results = yolo_model.track(frame_top, persist=True, iou=0.3, 
                                          tracker="bytetrack.yaml", verbose=False)[0]
                
                # debug
                # frame_top = results.plot()
                
                grid_status, _, grid_transform = process_yolo_results(results, GRID_ROWS, GRID_COLS, TRAY_CLASS_NAME, OBJECT_CLASS_NAMES)
                
                if grid_transform[0] is not None:
                    projected_grids = project_grid_to_head(
                        grid_transform, grid_status, GRID_ROWS, GRID_COLS,
                        top_proj, head_proj, rvec, tvec, ASSUMED_DEPTH
                    )
                    draw_grids(left_image, projected_grids)
                    process_objects(left_image, results, top_proj, head_proj, rvec, tvec, ASSUMED_DEPTH, OBJECT_CONFIGS)
                
                cv2.imshow('camera_head', left_image)
                # cv2.imshow('YOLO Detection', frame_top)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
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


if __name__ == "__main__":
    # 配置常量
    # TRAY_CLASS_NAME = 'sorting_tray'
    TRAY_CLASS_NAME = 'dish_drying_rack'
    OBJECT_CLASS_NAMES = ['cola', 'mug','dish']
    OBJECT_CONFIGS = {
        'cola': {'size': [0.06, 0.06, 0.16], 'color': (0, 255, 255)},
        'mug': {'size': [0.08, 0.08, 0.07], 'color': (255, 0, 255)},
        # 'dish': {'size': [0.20, 0.15, 0.02], 'color': (255, 255, 0)},
    }
    GRID_ROWS = 1
    GRID_COLS = 7
    ASSUMED_DEPTH = 1.50

    YOLO_MODEL_PATH = "weight/best_dish.pt"
    # Run
    main()