# 文件作用：定义检测结果、图像特征、伺服命令和控制模式的统一数据类型。
# 将检测器输出与控制算法解耦，并检查输入数据的维度、有效数值和置信度范围。
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ServoMode(str, Enum):
    IBVS = "ibvs"
    PBVS = "pbvs"
    HYBRID = "hybrid"


@dataclass(slots=True, frozen=True)
class Detection:
    """Detector-neutral target observation in image coordinates."""

    class_name: str
    confidence: float
    bbox_xyxy: np.ndarray
    centroid_px: np.ndarray
    stamp_s: float

    def __post_init__(self) -> None:
        bbox = np.asarray(self.bbox_xyxy, dtype=float).reshape(4)
        centroid = np.asarray(self.centroid_px, dtype=float).reshape(2)
        if not np.isfinite(bbox).all() or not np.isfinite(centroid).all():
            raise ValueError("detection coordinates must be finite")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")
        if not np.isfinite(self.stamp_s):
            raise ValueError("detection timestamp must be finite")
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "centroid_px", centroid)


@dataclass(slots=True, frozen=True)
class FeatureObservation:
    pixel: np.ndarray
    depth_m: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        pixel = np.asarray(self.pixel, dtype=float).reshape(2)
        if not np.isfinite(pixel).all():
            raise ValueError("feature pixel must be finite")
        if not np.isfinite(self.depth_m) or self.depth_m <= 0.0:
            raise ValueError("feature depth must be positive and finite")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("feature confidence must be in [0, 1]")
        object.__setattr__(self, "pixel", pixel)


@dataclass(slots=True, frozen=True)
class ServoCommand:
    linear_velocity_world: np.ndarray
    next_position_world: np.ndarray
    mode: ServoMode
    position_error_m: float
    image_error_px: float
    ibvs_weight: float

    def __post_init__(self) -> None:
        velocity = np.asarray(self.linear_velocity_world, dtype=float).reshape(3)
        position = np.asarray(self.next_position_world, dtype=float).reshape(3)
        if not np.isfinite(velocity).all() or not np.isfinite(position).all():
            raise ValueError("servo command must be finite")
        object.__setattr__(self, "linear_velocity_world", velocity)
        object.__setattr__(self, "next_position_world", position)
