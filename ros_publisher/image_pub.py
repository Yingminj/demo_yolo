# ROS2相关导入
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from tf2_ros import TransformListener, Buffer
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs
from scipy.spatial.transform import Rotation
import numpy as np

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
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 存储最新的左右手位姿
        self.left_hand_pose = None
        self.right_hand_pose = None
        
    def get_hand_poses(self):
        """获取左右手在head_left_camera坐标系下的位姿"""
        try:
            now = self.get_clock().now()
            timeout = rclpy.duration.Duration(seconds=0.1)

            # 获取left_tool相对于head_left_camera的变换
            left_transform = self.tf_buffer.lookup_transform(
                'head_left_camera',
                'left_tool',
                rclpy.time.Time()
            )
            
            # 获取right_tool相对于head_left_camera的变换
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
        
        # 提取旋转（四元数转旋转矩阵）
        quat = [
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w
        ]

        # 四元数 -> 欧拉角 (roll, pitch, yaw)
        r = Rotation.from_quat(quat)
        euler = r.as_euler('xyz', degrees=False)
        
        # 欧拉角 -> 四元数
        r_new = Rotation.from_euler('xyz', euler, degrees=False)
        quat_new = r_new.as_quat()
        
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
