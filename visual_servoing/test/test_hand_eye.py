# 文件作用：验证棋盘格尺寸约定、ROS 位姿转换以及眼在手上手眼标定的变换方向。
import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from visual_servoing.core.hand_eye import (
    CalibrationSample,
    invert_transform,
    make_board_points,
    pose_to_matrix,
    solve_hand_eye,
)


def _transform(rvec, translation):
    matrix = np.eye(4)
    matrix[:3, :3], _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def test_a3_board_uses_8_by_12_inner_corners():
    pattern_size, points = make_board_points(9, 13, 0.03)

    assert pattern_size == (8, 12)
    assert points.shape == (96, 3)
    np.testing.assert_allclose(points[-1], [0.21, 0.33, 0.0])


def test_pose_conversion_uses_ros_wxyz_order():
    pose = pose_to_matrix([1.0, 2.0, 3.0], [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])

    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose[:3, :3] @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(pose @ invert_transform(pose), np.eye(4), atol=1e-7)


def test_solver_returns_camera_to_ee_transform():
    pattern, board_points = make_board_points(9, 13, 0.03)
    del pattern
    camera_matrix = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]
    )
    distortion = np.zeros((5, 1))
    expected_ee_from_camera = _transform([0.15, -0.20, 0.08], [0.04, -0.03, 0.06])
    world_from_board = _transform([0.10, 0.05, -0.08], [0.32, 0.01, 0.18])
    samples = []

    robot_poses = [
        ([0.00, 0.00, 0.00], [0.00, 0.00, 0.00]),
        ([0.25, 0.00, 0.05], [0.03, -0.02, 0.00]),
        ([0.00, 0.25, 0.00], [-0.04, 0.02, 0.03]),
        ([-0.20, 0.10, 0.08], [0.05, 0.04, -0.03]),
        ([0.15, -0.20, 0.03], [-0.02, 0.06, 0.05]),
        ([-0.10, -0.15, 0.06], [0.06, -0.04, 0.02]),
        ([0.20, 0.18, -0.05], [-0.05, -0.03, 0.04]),
        ([-0.18, 0.20, -0.02], [0.02, 0.05, -0.06]),
    ]
    for rvec, translation in robot_poses:
        world_from_ee = _transform(rvec, translation)
        camera_from_board = (
            invert_transform(world_from_ee @ expected_ee_from_camera) @ world_from_board
        )
        board_rvec, _ = cv2.Rodrigues(camera_from_board[:3, :3])
        image_points, _ = cv2.projectPoints(
            board_points,
            board_rvec,
            camera_from_board[:3, 3],
            camera_matrix,
            distortion,
        )
        samples.append(
            CalibrationSample(
                world_from_ee=world_from_ee,
                camera_from_board=camera_from_board,
                image_points=image_points.reshape(-1, 2),
                reprojection_error_px=0.0,
            )
        )

    result = solve_hand_eye(samples, camera_matrix, distortion, board_points)

    np.testing.assert_allclose(
        result["T_ee_camera"], expected_ee_from_camera, atol=1e-5
    )
    assert result["quality"]["board_translation_rms_m"] < 1e-6
    assert result["quality"]["reprojection_rms_px"] < 1e-5
