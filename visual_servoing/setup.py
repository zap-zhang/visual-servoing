# 文件作用：配置 Python 包的安装，注册 ROS 节点可执行入口并安装参数、启动文件和包索引。
import os
from glob import glob
from setuptools import find_packages, setup


package_name = "visual_servoing"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ubuntu",
    maintainer_email="ubuntu@todo.todo",
    description="Extensible hybrid IBVS/PBVS visual servoing for the JetArm.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "hybrid_visual_servo = visual_servoing.ros_node:main",
            "hand_eye_calibration = visual_servoing.hand_eye_calibration:main",
        ],
    },
)
