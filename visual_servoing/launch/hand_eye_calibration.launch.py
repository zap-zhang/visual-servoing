# 文件作用：加载参数配置并启动棋盘格手眼标定节点。
# 相机、机械臂驱动和运动学服务需由其他启动文件先启动。
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("visual_servoing")
    default_config = os.path.join(package_share, "config", "hand_eye_calibration.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="visual_servoing",
                executable="hand_eye_calibration",
                name="hand_eye_calibration",
                output="screen",
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
