# 文件作用：提供视觉伺服所需的几何计算，包括深度投影、手眼坐标变换和四元数转换。
# 同时提供像素归一化、点特征平移交互矩阵及向量限幅工具。
from __future__ import annotations

import math

import numpy as np


def _finite_matrix(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def depth_to_camera_point(
    pixel: np.ndarray | tuple[float, float], depth_m: float, intrinsics: np.ndarray
) -> np.ndarray:
    """Project an RGB pixel and metric z depth into the optical camera frame."""

    u, v = _finite_matrix(pixel, (2,), "pixel")
    k = _finite_matrix(intrinsics, (3, 3), "intrinsics")
    z = float(depth_m)
    if z <= 0.0 or not math.isfinite(z):
        raise ValueError("depth must be positive and finite")
    fx, fy = k[0, 0], k[1, 1]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    return np.array([(u - k[0, 2]) * z / fx, (v - k[1, 2]) * z / fy, z])


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    matrix = _finite_matrix(transform, (4, 4), "transform")
    xyz = _finite_matrix(point, (3,), "point")
    homogeneous = matrix @ np.append(xyz, 1.0)
    if abs(homogeneous[3]) <= 1e-12:
        raise ValueError("invalid homogeneous point")
    return homogeneous[:3] / homogeneous[3]


def camera_point_to_world(
    point_camera: np.ndarray,
    world_from_ee: np.ndarray,
    ee_from_camera: np.ndarray,
) -> np.ndarray:
    """Apply the calibrated T_world_ee and T_ee_camera transforms."""

    return transform_point(world_from_ee @ ee_from_camera, point_camera)


def pose_to_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Build a homogeneous matrix from a ROS pose using w,x,y,z quaternion order."""

    xyz = _finite_matrix(position, (3,), "position")
    w, x, y, z = _finite_matrix(quaternion_wxyz, (4,), "quaternion")
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = np.array([w, x, y, z], dtype=float) / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = xyz
    return matrix


def clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    value = _finite_matrix(vector, (3,), "vector")
    limit = float(maximum)
    if limit <= 0.0 or not math.isfinite(limit):
        raise ValueError("maximum must be positive and finite")
    norm = float(np.linalg.norm(value))
    return value if norm <= limit else value * (limit / norm)


def normalized_pixel(pixel: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    uv = _finite_matrix(pixel, (2,), "pixel")
    k = _finite_matrix(intrinsics, (3, 3), "intrinsics")
    return np.array([(uv[0] - k[0, 2]) / k[0, 0], (uv[1] - k[1, 2]) / k[1, 1]])


def point_interaction_matrix(x: float, y: float, depth_m: float) -> np.ndarray:
    """Classical point-feature interaction matrix, restricted to translation."""

    z = float(depth_m)
    if z <= 0.0 or not math.isfinite(z):
        raise ValueError("interaction-matrix depth must be positive and finite")
    return np.array([[-1.0 / z, 0.0, x / z], [0.0, -1.0 / z, y / z]])
