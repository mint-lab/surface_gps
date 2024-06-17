from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config_file = DeclareLaunchArgument(
        "config_file",
        default_value="/home/mint/nrf_ws/src/surface_gps/config/config.yaml",
        description="Path to the config file",
    )

    return LaunchDescription(
        [
            config_file,
            Node(
                package="surface_gps",
                executable="localizer_node",
                name="localizer_node",
                output="screen",
                parameters=[{"config_file": LaunchConfiguration("config_file")}],
            ),
        ]
    )
