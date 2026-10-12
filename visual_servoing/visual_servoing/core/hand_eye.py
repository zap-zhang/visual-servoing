# 文件作用：提供棋盘格手眼标定的纯数学计算，包括棋盘格检测、PnP、OpenCV 手眼求解和质量评估。
# 本文件不依赖 ROS，便于在 Linux ROS 2 环境中运行，也便于离线单元测试。
"""Chessboard eye-in-hand calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass
class CalibrationSample:
    """One synchronized robot pose and chessboard observation."""

    world_from_ee: np.ndarray
    camera_from_board: np.ndarray
    image_points: np.ndarray
    reprojection_error_px: float
    image_path: str = ""


def make_board_points(
    squares_x: int, squares_y: int, square_size_m: float
) -> tuple[tuple[int, int], np.ndarray]:
    """Return OpenCV pattern size and planar board points.

    For a board printed with 9 x 13 complete squares, OpenCV sees 8 x 12
    inner corners. The first point is the top-left inner corner in the
    detected image order; the absolute board origin does not affect hand-eye
    calibration as long as the order is consistent.
    """

    if squares_x < 3 or squares_y < 3:
        raise ValueError("chessboard must contain at least 3 x 3 squares")
    if square_size_m <= 0.0 or not np.isfinite(square_size_m):
        raise ValueError("square_size_m must be positive and finite")

    pattern_size = (squares_x - 1, squares_y - 1)
    grid_x, grid_y = np.meshgrid(
        np.arange(pattern_size[0], dtype=np.float32),
        np.arange(pattern_size[1], dtype=np.float32),
    )
    points = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    points[:, 0] = grid_x.reshape(-1) * float(square_size_m)
    points[:, 1] = grid_y.reshape(-1) * float(square_size_m)
    return pattern_size, points


def detect_chessboard(
    image: np.ndarray, pattern_size: tuple[int, int]
) -> tuple[bool, np.ndarray | None]:
    """Detect and refine all chessboard inner corners in an image."""

    if image is None or image.size == 0:
        return False, None
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    corners: np.ndarray | None = None
    found = False
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    else:
        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK,
        )

    if not found or corners is None:
        return False, None

    # SB already returns sub-pixel corners, while the legacy detector does not.
    if not hasattr(cv2, "findChessboardCornersSB"):
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            40,
            1e-3,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, np.asarray(corners, dtype=np.float64).reshape(-1, 2)


def pose_to_matrix(position: Sequence[float], quaternion_wxyz: Sequence[float]) -> np.ndarray:
    """Build a homogeneous transform from ROS position and wxyz quaternion."""

    xyz = np.asarray(position, dtype=float).reshape(3)
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float).reshape(4)
    norm = float(np.linalg.norm([w, x, y, z]))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = np.asarray([w, x, y, z]) / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform."""

    matrix = np.asarray(transform, dtype=float).reshape(4, 4)
    result = np.eye(4, dtype=float)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Average translations and project the summed rotation onto SO(3)."""

    if not transforms:
        raise ValueError("at least one transform is required")
    rotations = np.sum([np.asarray(t)[:3, :3] for t in transforms], axis=0)
    u, _, vh = np.linalg.svd(rotations)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    result = np.eye(4, dtype=float)
    result[:3, :3] = rotation
    result[:3, 3] = np.mean([np.asarray(t)[:3, 3] for t in transforms], axis=0)
    return result


def _project_points(
    board_points: np.ndarray,
    camera_from_board: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(camera_from_board[:3, :3])
    projected, _ = cv2.projectPoints(
        np.asarray(board_points, dtype=np.float64),
        rvec,
        camera_from_board[:3, 3].reshape(3, 1),
        camera_matrix,
        distortion,
    )
    return projected.reshape(-1, 2)


def solve_hand_eye(
    samples: Sequence[CalibrationSample],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    board_points: np.ndarray,
    method: int = cv2.CALIB_HAND_EYE_TSAI,
) -> dict[str, Any]:
    """Solve eye-in-hand calibration and calculate fixed-board consistency.

    OpenCV expects:
      R_gripper2base = world_from_ee rotation
      R_target2cam = camera_from_board rotation

    For the eye-in-hand case the returned transform is camera-to-gripper,
    which is exactly T_ee_camera used by the visual-servoing package.
    """

    if len(samples) < 3:
        raise ValueError("at least 3 samples are required")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1, 1)

    r_gripper2base = [np.asarray(sample.world_from_ee)[:3, :3] for sample in samples]
    t_gripper2base = [
        np.asarray(sample.world_from_ee)[:3, 3].reshape(3, 1) for sample in samples
    ]
    r_target2cam = [np.asarray(sample.camera_from_board)[:3, :3] for sample in samples]
    t_target2cam = [
        np.asarray(sample.camera_from_board)[:3, 3].reshape(3, 1) for sample in samples
    ]

    rotation_cam2ee, translation_cam2ee = cv2.calibrateHandEye(
        r_gripper2base,
        t_gripper2base,
        r_target2cam,
        t_target2cam,
        method=method,
    )
    ee_from_camera = np.eye(4, dtype=float)
    ee_from_camera[:3, :3] = np.asarray(rotation_cam2ee, dtype=float).reshape(3, 3)
    ee_from_camera[:3, 3] = np.asarray(translation_cam2ee, dtype=float).reshape(3)
    camera_from_ee = invert_transform(ee_from_camera)

    board_poses_world = [
        np.asarray(sample.world_from_ee) @ ee_from_camera @ np.asarray(sample.camera_from_board)
        for sample in samples
    ]
    board_pose_world = average_transforms(board_poses_world)
    translation_errors = [
        float(np.linalg.norm(pose[:3, 3] - board_pose_world[:3, 3]))
        for pose in board_poses_world
    ]
    rotation_errors = [
        _rotation_angle_deg(board_pose_world[:3, :3].T @ pose[:3, :3])
        for pose in board_poses_world
    ]

    reprojection_errors: list[float] = []
    for sample in samples:
        camera_from_board_predicted = (
            invert_transform(np.asarray(sample.world_from_ee) @ ee_from_camera)
            @ board_pose_world
        )
        projected = _project_points(
            board_points, camera_from_board_predicted, camera_matrix, distortion
        )
        observed = np.asarray(sample.image_points, dtype=float).reshape(-1, 2)
        reprojection_errors.append(float(np.sqrt(np.mean(np.sum((projected - observed) ** 2, axis=1)))))

    pairwise_rotation_deg: list[float] = []
    pairwise_translation_m: list[float] = []
    for index, left in enumerate(samples):
        for right in samples[index + 1 :]:
            relative = invert_transform(left.world_from_ee) @ right.world_from_ee
            pairwise_rotation_deg.append(_rotation_angle_deg(relative[:3, :3]))
            pairwise_translation_m.append(float(np.linalg.norm(relative[:3, 3])))

    return {
        "T_ee_camera": ee_from_camera,
        "T_camera_ee": camera_from_ee,
        "T_world_board": board_pose_world,
        "board_poses_world": board_poses_world,
        "quality": {
            "sample_count": len(samples),
            "reprojection_rms_px": float(np.sqrt(np.mean(np.square(reprojection_errors)))),
            "reprojection_max_px": float(np.max(reprojection_errors)),
            "board_translation_rms_m": float(np.sqrt(np.mean(np.square(translation_errors)))),
            "board_translation_max_m": float(np.max(translation_errors)),
            "board_rotation_rms_deg": float(np.sqrt(np.mean(np.square(rotation_errors)))),
            "board_rotation_max_deg": float(np.max(rotation_errors)),
            "max_pairwise_rotation_deg": float(max(pairwise_rotation_deg, default=0.0)),
            "max_pairwise_translation_m": float(max(pairwise_translation_m, default=0.0)),
        },
    }


def matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    """Convert a matrix to YAML-safe nested Python lists."""

    return np.asarray(matrix, dtype=float).round(12).tolist()


def flatten_matrix(matrix: np.ndarray) -> list[float]:
    """Flatten a transform in row-major order for ROS parameters."""

    return np.asarray(matrix, dtype=float).reshape(4, 4).round(12).reshape(-1).tolist()
