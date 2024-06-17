import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path

from surface_gps.simple_localizer import SimpleLocalizer


class Localizer_node(Node):
    """A ROS node for the Exponential moving average (EMA) with GPS and IMU data(AHRS)"""

    def __init__(self):
        '''A constructor'''
        super().__init__("localizer_node")
        self.subscription = self.create_subscription(
            NavSatFix, "/ublox/fix", self.gps_callback, 10
        )

        self.imu_subscription = self.create_subscription(
            Imu, "/imu/data", self.imu_callback, 10
        )

        self.pub_tf = self.create_publisher(TFMessage, "/tf", 10)
        self.pub_robot_pose = self.create_publisher(PoseStamped, "/robot/pose", 10)
        self.pub_robot_path = self.create_publisher(Path, "/robot/path", 10)

        self.pose_list = []
        self.path_msg = Path()

        self.simple_localizer = SimpleLocalizer()

        # Initialize the pose attributes
        self.x, self.y, self.z = 0., 0., 0.
        self.qx, self.qy, self.qz, self.qw = 0., 0., 0., 1.

    def gps_callback(self, msg: NavSatFix):
        '''A callback function for the GPS data'''
        header = msg.header
        gps_data = (
            (msg.latitude, msg.longitude, msg.altitude),
            msg.position_covariance,
        )

        self.simple_localizer.apply_gps_data(gps_data)

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

        q_xyzw = self.simple_localizer.get_pose()[1]  # (qx, qy, qz, qw)
        self.qx, self.qy, self.qz, self.qw = q_xyzw

        # Publish TF
        _msg = TransformStamped()
        _msg.header.stamp = header.stamp
        _msg.header.frame_id = "odom"
        _msg.child_frame_id = "base_link"
        _msg.transform.translation.x = float(self.x)
        _msg.transform.translation.y = float(self.y)
        _msg.transform.translation.z = float(self.z)
        _msg.transform.rotation.x = float(self.qx)
        _msg.transform.rotation.y = float(self.qy)
        _msg.transform.rotation.z = float(self.qz)
        _msg.transform.rotation.w = float(self.qw)

        tf_msg = TFMessage()
        tf_msg.transforms = [_msg]
        self.pub_tf.publish(tf_msg)

        # Publish nav_path
        pose_msg = PoseStamped()
        pose_msg.header.stamp = header.stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = float(self.x)
        pose_msg.pose.position.y = float(self.y)
        pose_msg.pose.position.z = float(self.z)
        pose_msg.pose.orientation.x = float(self.qx)
        pose_msg.pose.orientation.y = float(self.qy)
        pose_msg.pose.orientation.z = float(self.qz)
        pose_msg.pose.orientation.w = float(self.qw)

        self.pose_list.append(pose_msg)
        self.path_msg.header = pose_msg.header
        self.path_msg.poses = self.pose_list

        # Publish pose and path
        self.pub_robot_pose.publish(pose_msg)
        self.pub_robot_path.publish(self.path_msg)


def main():
    rclpy.init()
    node = Localizer_node()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
