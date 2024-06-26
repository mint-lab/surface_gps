import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped, PoseStamped
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import NavSatFix, Imu
from nav_msgs.msg import Path
from std_msgs.msg import UInt8, Float32MultiArray

from surface_gps_interface.srv import AvgGPS
from surface_gps.simple_localizer import SimpleLocalizer


class Localizer_node(Node):
    """A ROS node for the Exponential moving average (EMA) with GPS and IMU data(AHRS)"""

    def __init__(self):
        """A constructor"""
        super().__init__("localizer_node")
        self.gps_subscription = self.create_subscription(
            NavSatFix, "/ublox/fix", self.gps_callback, 10
        )

        self.imu_subscription = self.create_subscription(
            Imu, "/imu/data", self.imu_callback, 10
        )

        self.gps_avg_srv = self.create_service(
            AvgGPS, "/gps_avg", self.gps_avg_callback
        )

        self.pub_tf = self.create_publisher(TFMessage, "/tf", 10)
        self.pub_robot_pose = self.create_publisher(PoseStamped, "/surface_gps/pose", 10)
        self.pub_robot_path = self.create_publisher(Path, "/surface_gps/path", 10)
        self.pub_latlon = self.create_publisher(NavSatFix, "/surface_gps/latlon", 10)

        self.pose_list = []
        self.path_msg = Path()

        self.simple_localizer = SimpleLocalizer()

        self.config_file = (
            self.declare_parameter("config_file").get_parameter_value().string_value
        )
        self.initialize()

    def initialize(self):
        """Initialize the simple localizer"""
        # Initialize the pose attributes
        self.x, self.y, self.z = 0.0, 0.0, 0.0
        self.qx, self.qy, self.qz, self.qw = 0.0, 0.0, 0.0, 1.0

        # Load the configuration file
        if self.config_file:
            if self.simple_localizer.load_config_file(self.config_file):
                self.get_logger().info(
                    f"Loaded the configuration file: {self.config_file}"
                )
                self.get_logger().info(
                    f"Configuration: \n{self.simple_localizer.get_config()}"
                )
            else:
                self.get_logger().error(
                    f"Failed to load the configuration file: {self.config_file}"
                )
        else:
            self.get_logger().warn("No configuration file is loaded")

        return True

    def gps_callback(self, msg: NavSatFix):
        """A callback function for the GPS data"""
        header = msg.header
        gps_data = (
            (msg.latitude, msg.longitude, msg.altitude),
            msg.position_covariance,
        )

        self.simple_localizer.apply_gps_data(gps_data)

        p_xyz = self.simple_localizer.get_pose()[0]  # (x, y, z)
        self.x, self.y, self.z = p_xyz

    def imu_callback(self, msg: Imu):
        """A callback function for the IMU data(AHRS)"""
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

        self.pub_robot_pose.publish(pose_msg)
        self.pub_robot_path.publish(self.path_msg)

        # Publish latlon
        latlon_msg = NavSatFix()
        latlon_msg.header.stamp = header.stamp
        latlon_msg.header.frame_id = "map"
        latlonalt = self.simple_localizer.get_gps_position()

        if len(latlonalt) == 3:
            latlon_msg.latitude = latlonalt[0]
            latlon_msg.longitude = latlonalt[1]
            latlon_msg.altitude = latlonalt[2]
            latlon_msg.position_covariance = [0.0] * 9

            self.pub_latlon.publish(latlon_msg)

    def gps_avg_callback(self, request: UInt8, response: Float32MultiArray):
        """A service callback function for the GPS averaging"""
        # TODO: Handle unexpected behavior when the GPS data is not available
        gps_latlon_list = []
        gps_latlon = None
        gps_position_list = []
        gps_position = None
        for _ in range(request.filter_size):
            gps_latlon = self.simple_localizer.get_gps_position()
            gps_position = self.simple_localizer.get_pose()[0].tolist()

            if gps_latlon and gps_position:
                gps_latlon_list.append(gps_latlon)
                gps_position_list.append(gps_position)

        if not gps_latlon_list:
            response_data = Float32MultiArray(data=[])
            response.latlon = response_data
            self.get_logger().warn("No GPS data is available")
            return response
        if not gps_position_list:
            response_data = Float32MultiArray(data=[])
            response.position = response_data
            self.get_logger().warn("No GPS data is available")
            return response

        # [lat, lon, delta_h]
        gps_latlon_avg = np.sum(gps_latlon_list, axis=0) / len(gps_latlon_list)
        response_data = Float32MultiArray()
        response_data.data = gps_latlon_avg.tolist()
        response.latlon = response_data

        gps_origin = self.simple_localizer.get_gps_origin()
        response_origin = Float32MultiArray()
        response_origin.data = gps_origin
        response.origin = response_origin

        gps_position_avg = np.sum(gps_position_list, axis=0) / len(gps_position_list)
        response_pose = Float32MultiArray()
        response_pose.data = gps_position_avg.tolist()
        response.position = response_pose

        self.get_logger().info(f"GPS origin: {response.origin}")
        self.get_logger().info(f"GPS latlon: {response.latlon}")
        self.get_logger().info(f"GPS xyz: {response.position}")
        return response


def main():
    rclpy.init()
    node = Localizer_node()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
