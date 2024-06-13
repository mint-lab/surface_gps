import numpy as np
import transforms3d.quaternions as quat

from tf_transformations import euler_from_quaternion
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise

import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path

from numpy.linalg import inv
from scipy.spatial.transform import Rotation
from pyproj import Transformer


class EKF_node(Node):
    def __init__(self):
        super().__init__('ekf_node')
        self.subscription = self.create_subscription(
            NavSatFix,
            '/ublox/fix',
            self.fix_callback,
            10)
        
        self.imu_subscription = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )
        self.pub_tf = self.create_publisher(TFMessage, '/tf', 10)
        self.pub_path = self.create_publisher(Path, '/ekf_path', 10)
        self.pub_nav_path = self.create_publisher(Path, '/nav_path', 10)

        self.pub_ekf_pose = self.create_publisher(PoseStamped, '/ekf_pose', 10)
        self.pub_nav_pose = self.create_publisher(PoseStamped, '/nav_pose', 10)

        self.path_list = []
        self.path_msg = Path()

        self.nav_list = []
        self.nav_msg = Path()

        self.positions = np.empty((0,3))
        self.init_east = 0
        self.init_north = 0
        self.init_alt = 0
        self.relative_matrix = 0
        self.is_fix_init = True
        self.is_imu_init = True

        self.init_quart = np.zeros(4)

        self.localizer = ExtendedKalmanFilter(dim_x=7, dim_z=7)
        self.localizer.F = np.eye(7)
        self.localizer.Q = np.eye(7)*0.01
        self.h = lambda x: x
        self.H = lambda x: np.eye(7)
        self.R = np.zeros((7,7))

        self.x = 0
        self.y = 0
        self.z = 0

    def fix_callback(self, data):
        if (self.is_imu_init == False):
            latitude = data.latitude
            longitude = data.longitude
            altitude = data.altitude

            self.R[0,0] = data.position_covariance[0]
            self.R[1,1] = data.position_covariance[4]
            self.R[2,2] = data.position_covariance[8]

            utm_e, utm_n = self.lat_lon_to_utm(latitude, longitude)

            # Set Initial Position
            if (self.is_fix_init == True):
                self.init_east = utm_e
                self.init_north = utm_n
                self.init_alt = altitude
                self.is_fix_init=False

            # Compute Relative Position (N, W, Z) axis
            diff_e = -(utm_e-self.init_east)
            diff_n = utm_n-self.init_north
            diff_a = altitude-self.init_alt
            pos_array = np.array([diff_n, diff_e, diff_a])

            # Change (N, W, Z) axis to (X, Y, Z) axis
            filtered_position = inv(self.relative_matrix) @ pos_array
            # print(filtered_position)
            self.x = filtered_position[0]
            self.y = filtered_position[1]
            self.z = filtered_position[2]

    def lat_lon_to_utm(self, latitude, longitude):
        transformer = Transformer.from_crs('EPSG:4326', 'EPSG:32752')  # EPSG:4326 = GPS 기준 좌표계, EPSG:32752 = 서울(52S) 기준 좌표계
        utm_easting, utm_northing = transformer.transform(latitude, longitude)
        return utm_easting, utm_northing

    def imu_callback(self, data):
        q_x = data.orientation.x
        q_y = data.orientation.y
        q_z = data.orientation.z
        q_w = data.orientation.w

        self.R[3,3] = data.orientation_covariance[0]
        self.R[4,4] = data.orientation_covariance[4]
        self.R[5,5] = data.orientation_covariance[8]

        # Set Initial Heading
        if (self.is_imu_init == True):
            self.init_quart = np.array([q_w, q_x, q_y, q_z])
            self.relative_matrix = Rotation.from_quat([q_x, q_y, q_z, q_w]).as_matrix()
            # print(self.relative_matrix)
            self.is_imu_init=False

        # Compute Relative Orientaion
        current_q = np.array([q_w, q_x, q_y, q_z])
        delta_q = quat.qmult(quat.qinverse(self.init_quart), current_q)
        self.ox = delta_q[1]
        self.oy = delta_q[2]
        self.oz = delta_q[3]
        self.ow = delta_q[0]

        # Predict & Update Position
        z = np.array([[self.x, self.y, self.z, self.ox, self.oy, self.oz, self.ow]]).T
        self.localizer.update(z, HJacobian=self.H, Hx=self.h)
        self.localizer.predict()
        # print(self.localizer.x)


        # Publish TF
        msg = TransformStamped()
        msg.header.stamp = data.header.stamp
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.transform.rotation.x = float(self.localizer.x[3])
        msg.transform.rotation.y = float(self.localizer.x[4])
        msg.transform.rotation.z = float(self.localizer.x[5])
        msg.transform.rotation.w = float(self.localizer.x[6])
        msg.transform.translation.x = float(self.localizer.x[0])
        msg.transform.translation.y = float(self.localizer.x[1])
        msg.transform.translation.z = float(self.localizer.x[2])

        tf_msg = TFMessage()
        tf_msg.transforms = [msg]
        self.pub_tf.publish(tf_msg)

        # Publish EKF_path
        pose_msg = PoseStamped()
        pose_msg.header.stamp = data.header.stamp
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(self.localizer.x[0])
        pose_msg.pose.position.y = float(self.localizer.x[1])
        pose_msg.pose.position.z = float(self.localizer.x[2])
        pose_msg.pose.orientation.x = float(self.localizer.x[3])
        pose_msg.pose.orientation.y = float(self.localizer.x[4])
        pose_msg.pose.orientation.z = float(self.localizer.x[5])
        pose_msg.pose.orientation.w = float(self.localizer.x[6])

        self.path_list.append(pose_msg)
        # print(self.path_list)
        self.path_msg.header = pose_msg.header
        self.path_msg.poses = self.path_list
        self.pub_path.publish(self.path_msg)

        # Publish EKF_pose
        self.pub_ekf_pose.publish(pose_msg)

        nav_msg = PoseStamped()
        nav_msg.header.stamp = data.header.stamp
        nav_msg.header.frame_id = 'map'
        nav_msg.pose.position.x = float(self.x)
        nav_msg.pose.position.y = float(self.y)
        nav_msg.pose.position.z = float(self.z)
        nav_msg.pose.orientation.x = float(self.ox)
        nav_msg.pose.orientation.y = float(self.oy)
        nav_msg.pose.orientation.z = float(self.oz)
        nav_msg.pose.orientation.w = float(self.ow)

        self.nav_list.append(nav_msg)
        # print(self.path_list)
        self.nav_msg.header = pose_msg.header
        self.nav_msg.poses = self.nav_list
        self.pub_nav_path.publish(self.nav_msg)

        # Publish Nav_pose
        self.pub_nav_pose.publish(nav_msg)

def main():
    rclpy.init()
    node = EKF_node()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
