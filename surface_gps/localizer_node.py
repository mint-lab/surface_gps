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

from surface_gps.simple_localizer import SimpleLocalizer


class EKF_node(Node):
    """A ROS node for the Extended Kalman Filter (EKF) with GPS and IMU data(AHRS)"""

    def __init__(self):
        '''A constructor'''
        super().__init__("ekf_node")
        self.subscription = self.create_subscription(
            NavSatFix, "/ublox/fix", self.gps_callback, 10
        )

        self.imu_subscription = self.create_subscription(
            Imu, "/imu/data", self.imu_callback, 10
        )

        self.pub_tf = self.create_publisher(TFMessage, "/tf", 10)
        self.pub_path = self.create_publisher(Path, "/ekf_path", 10)
        self.pub_nav_path = self.create_publisher(Path, "/nav_path", 10)

        self.pub_ekf_pose = self.create_publisher(PoseStamped, "/ekf_pose", 10)
        self.pub_nav_pose = self.create_publisher(PoseStamped, "/nav_pose", 10)

        self.path_list = []
        self.path_msg = Path()

        self.nav_list = []
        self.nav_msg = Path()

        self.simple_localizer = SimpleLocalizer()

        self.ekf = ExtendedKalmanFilter(dim_x=7, dim_z=7)
        self.ekf.F = np.eye(7)
        self.ekf.Q = np.eye(7) * 0.01
        self.h = lambda x: x
        self.H = lambda x: np.eye(7)
        self.R = np.zeros((7, 7))

    def gps_callback(self, msg: NavSatFix):
        '''A callback function for the GPS data'''
        header = msg.header
        gps_data = (
            (msg.latitude, msg.longitude, msg.altitude),
            msg.position_covariance,
        )

        self.simple_localizer.apply_gps_data(gps_data)
        self.R[0, 0] = gps_data[1][0]  # sigma_xx
        self.R[1, 1] = gps_data[1][4]  # sigma_yy
        self.R[2, 2] = gps_data[1][8]  # sigma_zz

        p_xyz = self.simple_localizer.get_pose()[0]  # (x, y, z)
        self.x, self.y, self.z = p_xyz

    def imu_callback(self, msg: Imu):
        '''A callback function for the IMU data(AHRS)'''
        header = msg.header
        imu_data = (
            (
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ),
            msg.orientation_covariance,
        )

        self.simple_localizer.apply_ahrs_data(imu_data)
        self.R[3, 3] = imu_data[1][0]  # sigma_qxqx
        self.R[4, 4] = imu_data[1][4]  # sigma_qyqy
        self.R[5, 5] = imu_data[1][8]  # sigma_qzqz

        q_xyzw = self.simple_localizer.get_pose()[1]  # (qx, qy, qz, qw)
        self.qx, self.qy, self.qz, self.qw = q_xyzw

        # Predict and update pose
        state = np.array(
            [[self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw]]
        ).T
        self.ekf.update(state, HJacobian=self.H, Hx=self.h)
        self.ekf.predict()

        # Publish TF
        msg = TransformStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        msg.transform.translation.x = float(self.ekf.x[0])
        msg.transform.translation.y = float(self.ekf.x[1])
        msg.transform.translation.z = float(self.ekf.x[2])
        msg.transform.rotation.x = float(self.ekf.x[3])
        msg.transform.rotation.y = float(self.ekf.x[4])
        msg.transform.rotation.z = float(self.ekf.x[5])
        msg.transform.rotation.w = float(self.ekf.x[6])

        tf_msg = TFMessage()
        tf_msg.transforms = [msg]
        self.pub_tf.publish(tf_msg)

        # Publish EKF_path
        pose_msg = PoseStamped()
        pose_msg.header.stamp = header.stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(self.ekf.x[0])
        pose_msg.pose.position.y = float(self.ekf.x[1])
        pose_msg.pose.position.z = float(self.ekf.x[2])
        pose_msg.pose.orientation.x = float(self.ekf.x[3])
        pose_msg.pose.orientation.y = float(self.ekf.x[4])
        pose_msg.pose.orientation.z = float(self.ekf.x[5])
        pose_msg.pose.orientation.w = float(self.ekf.x[6])

        self.path_list.append(pose_msg)
        # print(self.path_list)
        self.path_msg.header = pose_msg.header
        self.path_msg.poses = self.path_list
        self.pub_path.publish(self.path_msg)

        # Publish EKF_pose
        self.pub_ekf_pose.publish(pose_msg)

        nav_msg = PoseStamped()
        nav_msg.header.stamp = header.stamp
        nav_msg.header.frame_id = "map"
        nav_msg.pose.position.x = float(self.x)
        nav_msg.pose.position.y = float(self.y)
        nav_msg.pose.position.z = float(self.z)
        nav_msg.pose.orientation.x = float(self.qx)
        nav_msg.pose.orientation.y = float(self.qy)
        nav_msg.pose.orientation.z = float(self.qz)
        nav_msg.pose.orientation.w = float(self.qw)

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


if __name__ == "__main__":
    main()
