# 文件作用：从目标检测框对应的深度区域中估计以米为单位的目标深度。
# 过滤无效值和两端异常值后取中位数；输入深度图需与检测图像像素对齐。
from __future__ import annotations

import math

import numpy as np


class DepthEstimator:
    """Robust metric-depth extraction from a detection region."""

    def __init__(self, scale_m_per_unit: float = 0.001, min_depth_m: float = 0.05):
        if scale_m_per_unit <= 0.0 or not math.isfinite(scale_m_per_unit):
            raise ValueError("scale_m_per_unit must be positive and finite")
        self.scale_m_per_unit = float(scale_m_per_unit)
        self.min_depth_m = float(min_depth_m)

    def estimate(self, depth_image: np.ndarray, bbox_xyxy: np.ndarray) -> float | None:
        depth = np.asarray(depth_image)
        bbox = np.asarray(bbox_xyxy, dtype=float).reshape(4)
        if depth.ndim != 2 or not np.isfinite(bbox).all():
            return None
        height, width = depth.shape
        x1, y1, x2, y2 = bbox
        left = max(0, min(width - 1, int(np.floor(x1))))
        top = max(0, min(height - 1, int(np.floor(y1))))
        right = max(left + 1, min(width, int(np.ceil(x2))))
        bottom = max(top + 1, min(height, int(np.ceil(y2))))
        roi = depth[top:bottom, left:right].astype(float) * self.scale_m_per_unit
        valid = roi[np.isfinite(roi) & (roi >= self.min_depth_m)]
        if valid.size == 0:
            return None
        lower, upper = np.percentile(valid, [10.0, 90.0])
        trimmed = valid[(valid >= lower) & (valid <= upper)]
        return float(np.median(trimmed if trimmed.size else valid))
