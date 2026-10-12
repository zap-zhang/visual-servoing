# 文件作用：定义 Python 包的公共入口，导出控制器、数据类型和采摘状态机接口。
# 便于其他模块复用本包算法，不会在导入时启动 ROS 节点。
"""Reusable hybrid visual-servoing components for the JetArm."""

from .core.controller import HybridVisualServo, ServoConfig
from .core.types import Detection, FeatureObservation, ServoMode
from .task.pick_state_machine import PickConfig, PickPhase, PickStateMachine

__all__ = [
    "Detection",
    "FeatureObservation",
    "HybridVisualServo",
    "ServoConfig",
    "ServoMode",
    "PickConfig",
    "PickPhase",
    "PickStateMachine",
]
