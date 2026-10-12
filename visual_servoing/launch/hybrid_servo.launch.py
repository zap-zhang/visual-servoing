# 文件作用：加载参数配置并启动混合视觉伺服 ROS2 节点，支持通过 config 参数指定配置文件。
# 本启动文件不负责启动相机、外部检测器、运动学服务或底层机械臂驱动。
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("visual_servoing")
    default_config = os.path.join(package_share, "config", "hybrid_servo.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="visual_servoing",
                executable="hybrid_visual_servo",
                name="hybrid_visual_servo",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
