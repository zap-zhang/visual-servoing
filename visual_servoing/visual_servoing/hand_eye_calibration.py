# 文件作用：运行 JetArm 眼在手上棋盘格手眼标定，采集机械臂末端位姿与相机图像，输出 T_ee_camera。
# 默认标定板为 A3 纸、9 x 13 个完整方格、单格 30 mm，对应 OpenCV 的 8 x 12 个内角点。
# 节点不会修改原有 transform.yaml，只生成独立结果文件，便于人工核对后复制参数。
"""ROS 2 eye-in-hand chessboard calibration node."""

from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .core.hand_eye import (
    CalibrationSample,
    detect_chessboard,
    flatten_matrix,
    make_board_points,
    matrix_to_list,
    pose_to_matrix,
    solve_hand_eye,
)


DEFAULT_AUTO_POSES = [
    # x, y, z, pitch(deg). Example values only; validate against the real setup.
    0.19,
    -0.08,
    0.22,
    -82.0,
    0.21,
    -0.08,
    0.22,
    -90.0,
    0.23,
    -0.08,
    0.22,
    -98.0,
    0.19,
    0.00,
    0.20,
    -82.0,
    0.21,
    0.00,
    0.20,
    -90.0,
    0.23,
    0.00,
    0.20,
    -98.0,
    0.19,
    0.08,
    0.22,
    -82.0,
    0.21,
    0.08,
    0.22,
    -90.0,
    0.23,
    0.08,
    0.22,
    -98.0,
    0.19,
    -0.06,
    0.27,
    -88.0,
    0.23,
    -0.06,
    0.27,
    -96.0,
    0.19,
    0.06,
    0.27,
    -88.0,
    0.23,
    0.06,
    0.27,
    -96.0,
]


class HandEyeCalibrationNode(Node):
    """Collect and solve a fixed-board eye-in-hand calibration."""

    def __init__(self) -> None:
        super().__init__("hand_eye_calibration")
        self._declare_parameters()

        self._bridge = CvBridge()
        self._latest_image: np.ndarray | None = None
        self._latest_image_time = 0.0
        self._latest_image_header = None
        self._camera_matrix: np.ndarray | None = None
        self._distortion = np.zeros((5, 1), dtype=np.float64)
        self._samples: list[CalibrationSample] = []
        self._capture_image: np.ndarray | None = None
        self._capture_points: np.ndarray | None = None
        self._capture_camera_from_board: np.ndarray | None = None
        self._capture_reprojection_error = 0.0
        self._pending_future = None
        self._pending_kind = ""
        self._auto_active = False
        self._auto_state = "idle"
        self._auto_index = 0
        self._manual_capture_pending = False
        self._last_debug_time = 0.0

        self._pattern_size, self._board_points = make_board_points(
            int(self.get_parameter("board_squares_x").value),
            int(self.get_parameter("board_squares_y").value),
            float(self.get_parameter("square_size_m").value),
        )
        self._output_file = self._expand_path(
            str(self.get_parameter("output_file").value)
        )
        self._sample_dir = self._expand_path(
            str(self.get_parameter("sample_dir").value)
        )
        self._method_name, self._method = self._hand_eye_method(
            str(self.get_parameter("method").value)
        )
        self._auto_poses = self._read_auto_poses()

        image_topic = str(self.get_parameter("image_topic").value)
        info_topic = str(self.get_parameter("camera_info_topic").value)
        self._image_pub = self.create_publisher(Image, "~/image_result", 2)
        self._status_pub = self.create_publisher(String, "~/status", 2)
        self._servo_pub = self.create_publisher(ServosPosition, "/servo_controller", 1)
        self._pose_client = self.create_client(
            GetRobotPose, str(self.get_parameter("get_pose_service").value)
        )
        self._ik_client = self.create_client(
            SetRobotPose, str(self.get_parameter("set_pose_service").value)
        )
        self.create_subscription(Image, image_topic, self._image_callback, 2)
        self.create_subscription(CameraInfo, info_topic, self._camera_info_callback, 2)
        self.create_service(Trigger, "~/start_auto", self._start_auto_callback)
        self.create_service(Trigger, "~/capture", self._capture_callback)
        self.create_service(Trigger, "~/solve", self._solve_callback)
        self.create_service(Trigger, "~/stop", self._stop_callback)
        self.create_service(Trigger, "~/reset", self._reset_callback)
        self.create_timer(0.05, self._timer_callback)

        self.get_logger().info(
            "棋盘格手眼标定节点已启动: "
            f"pattern={self._pattern_size[0]}x{self._pattern_size[1]} inner corners, "
            f"square={float(self.get_parameter('square_size_m').value):.3f} m, "
            f"method={self._method_name}"
        )
        self._publish_status("等待调用 ~/capture 或 ~/start_auto")

    def _declare_parameters(self) -> None:
        self.declare_parameter("image_topic", "/depth_cam/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/depth_cam/rgb/camera_info")
        self.declare_parameter("get_pose_service", "/kinematics/get_current_pose")
        self.declare_parameter("set_pose_service", "/kinematics/set_pose_target")
        self.declare_parameter(
            "output_file", "/home/ubuntu/ros2_ws/src/app/config/hand_eye_result.yaml"
        )
        self.declare_parameter(
            "sample_dir", "/home/ubuntu/ros2_ws/src/app/config/hand_eye_samples"
        )
        self.declare_parameter("board_squares_x", 9)
        self.declare_parameter("board_squares_y", 13)
        self.declare_parameter("square_size_m", 0.03)
        self.declare_parameter("minimum_samples", 12)
        self.declare_parameter("target_samples", 20)
        self.declare_parameter("settle_time_s", 2.0)
        self.declare_parameter("max_image_age_s", 0.5)
        self.declare_parameter("max_single_view_error_px", 4.0)
        self.declare_parameter("duplicate_translation_m", 0.004)
        self.declare_parameter("duplicate_rotation_deg", 2.0)
        self.declare_parameter("ik_duration_s", 1.0)
        self.declare_parameter("ik_pitch_range_deg", 6.0)
        self.declare_parameter("ik_resolution_deg", 1.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("method", "TSAI")
        self.declare_parameter("auto_pose_list", DEFAULT_AUTO_POSES)

    @staticmethod
    def _expand_path(value: str) -> str:
        return os.path.abspath(os.path.expanduser(value))

    @staticmethod
    def _hand_eye_method(name: str) -> tuple[str, int]:
        methods = {
            "TSAI": ("TSAI", cv2.CALIB_HAND_EYE_TSAI),
            "PARK": ("PARK", cv2.CALIB_HAND_EYE_PARK),
            "HORAUD": ("HORAUD", cv2.CALIB_HAND_EYE_HORAUD),
            "DANIILIDIS": ("DANIILIDIS", cv2.CALIB_HAND_EYE_DANIILIDIS),
            "ANDREFF": ("ANDREFF", cv2.CALIB_HAND_EYE_ANDREFF),
        }
        key = name.strip().upper()
        if key not in methods:
            raise ValueError(f"unsupported hand-eye method: {name}")
        return methods[key]

    def _read_auto_poses(self) -> list[tuple[float, float, float, float]]:
        values = [float(value) for value in self.get_parameter("auto_pose_list").value]
        if len(values) < 4 or len(values) % 4 != 0:
            raise ValueError("auto_pose_list must contain x,y,z,pitch groups")
        return [tuple(values[index : index + 4]) for index in range(0, len(values), 4)]

    def _image_callback(self, message: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:  # cv_bridge reports camera encoding errors here.
            self.get_logger().error(f"图像转换失败: {exc}")
            return
        self._latest_image = np.asarray(image, dtype=np.uint8).copy()
        self._latest_image_time = time.monotonic()
        self._latest_image_header = message.header

        if self.get_parameter("publish_debug_image").value and (
            self._auto_active or self._manual_capture_pending
        ):
            found, corners = detect_chessboard(self._latest_image, self._pattern_size)
            self._publish_debug(self._latest_image, corners if found else None, found)

    def _camera_info_callback(self, message: CameraInfo) -> None:
        matrix = np.asarray(message.k, dtype=np.float64).reshape(3, 3)
        if np.all(np.isfinite(matrix)) and matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0:
            self._camera_matrix = matrix
            distortion = np.asarray(message.d, dtype=np.float64).reshape(-1, 1)
            self._distortion = distortion if distortion.size else np.zeros((5, 1))

    def _publish_debug(
        self, image: np.ndarray, corners: np.ndarray | None, found: bool
    ) -> None:
        if not bool(self.get_parameter("publish_debug_image").value):
            return
        now = time.monotonic()
        if now - self._last_debug_time < 0.15:
            return
        self._last_debug_time = now
        debug = image.copy()
        if corners is not None:
            cv2.drawChessboardCorners(debug, self._pattern_size, corners.reshape(-1, 1, 2), found)
        label = "chessboard: OK" if found else "chessboard: NOT FOUND"
        cv2.putText(debug, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0) if found else (0, 0, 255), 2)
        message = self._bridge.cv2_to_imgmsg(debug, encoding="bgr8")
        if self._latest_image_header is not None:
            message.header = self._latest_image_header
        self._image_pub.publish(message)

    def _timer_callback(self) -> None:
        if self._pending_future is not None:
            if not self._pending_future.done():
                return
            future = self._pending_future
            kind = self._pending_kind
            self._pending_future = None
            self._pending_kind = ""
            if kind == "ik":
                self._finish_ik_future(future)
            elif kind == "pose":
                self._finish_pose_future(future)
            return

        if self._manual_capture_pending:
            self._manual_capture_pending = False
            self._begin_capture()
            return

        if self._auto_state == "move_request":
            self._begin_auto_move()
        elif self._auto_state == "settle":
            if time.monotonic() >= self._settle_deadline:
                self._begin_capture()

    def _start_auto_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._auto_active or self._pending_future is not None or self._manual_capture_pending:
            response.success = False
            response.message = "已有采集任务正在进行"
            return response
        if not self._auto_poses:
            response.success = False
            response.message = "auto_pose_list 为空"
            return response
        self._auto_active = True
        self._auto_state = "move_request"
        self._auto_index = 0
        self._publish_status("自动采集已开始，机械臂将按姿态列表移动")
        response.success = True
        response.message = f"开始自动采集，共 {len(self._auto_poses)} 个姿态"
        return response

    def _capture_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._auto_active or self._pending_future is not None or self._manual_capture_pending:
            response.success = False
            response.message = "当前正在采集，请等待或调用 ~/stop"
            return response
        if self._camera_matrix is None:
            response.success = False
            response.message = "尚未收到 camera_info"
            return response
        self._manual_capture_pending = True
        self._publish_status("已请求手动采集，请保持机械臂静止")
        response.success = True
        response.message = "采集请求已接受"
        return response

    def _solve_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._auto_active or self._pending_future is not None:
            response.success = False
            response.message = "采集尚未结束"
            return response
        success, message = self._solve_and_write()
        response.success = success
        response.message = message
        return response

    def _stop_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._auto_active = False
        self._auto_state = "idle"
        self._manual_capture_pending = False
        if self._pending_future is not None:
            self._pending_future.cancel()
        self._pending_future = None
        self._pending_kind = ""
        self._publish_status("采集已停止；机械臂当前动作不会被强制反向打断")
        response.success = True
        response.message = "采集已停止"
        return response

    def _reset_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self._auto_active or self._pending_future is not None:
            response.success = False
            response.message = "请先调用 ~/stop"
            return response
        self._samples.clear()
        self._publish_status("内存中的样本已清空，已生成的结果文件未删除")
        response.success = True
        response.message = "样本已清空"
        return response

    def _begin_auto_move(self) -> None:
        if not self._ik_client.service_is_ready():
            self.get_logger().warning("等待 /kinematics/set_pose_target 服务")
            return
        if self._auto_index >= len(self._auto_poses):
            self._finish_auto()
            return

        x, y, z, pitch = self._auto_poses[self._auto_index]
        request = SetRobotPose.Request()
        request.position = [x, y, z]
        request.pitch = pitch
        range_width = float(self.get_parameter("ik_pitch_range_deg").value)
        request.pitch_range = [pitch - range_width, pitch + range_width]
        request.resolution = float(self.get_parameter("ik_resolution_deg").value)
        request.duration = float(self.get_parameter("ik_duration_s").value)
        self._pending_future = self._ik_client.call_async(request)
        self._pending_kind = "ik"
        self._auto_state = "move_wait"
        self._publish_status(
            f"移动到第 {self._auto_index + 1}/{len(self._auto_poses)} 个姿态: "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}, pitch={pitch:.1f}"
        )

    def _finish_ik_future(self, future: Any) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._capture_failed(f"IK 服务失败: {exc}")
            return
        if not result.success or len(result.pulse) < 5:
            self._capture_failed("IK 没有返回有效舵机脉宽")
            return
        pulse = list(result.pulse)
        duration = float(self.get_parameter("ik_duration_s").value)
        set_servo_position(
            self._servo_pub,
            duration,
            ((10, 500), (5, 500), (4, pulse[3]), (3, pulse[2]), (2, pulse[1]), (1, pulse[0])),
        )
        self._settle_deadline = time.monotonic() + float(
            self.get_parameter("settle_time_s").value
        )
        self._auto_state = "settle"

    def _begin_capture(self) -> None:
        if self._latest_image is None or self._camera_matrix is None:
            self._capture_failed("尚未收到图像或 camera_info")
            return
        age = time.monotonic() - self._latest_image_time
        if age > float(self.get_parameter("max_image_age_s").value):
            self._capture_failed(f"最新图像过旧: {age:.2f}s")
            return

        image = self._latest_image.copy()
        found, corners = detect_chessboard(image, self._pattern_size)
        self._publish_debug(image, corners if found else None, found)
        if not found or corners is None:
            self._capture_failed("没有检测到完整棋盘格，当前姿态不计入样本")
            return

        try:
            ok, rvec, tvec = cv2.solvePnP(
                self._board_points,
                corners,
                self._camera_matrix,
                self._distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error as exc:
            self._capture_failed(f"棋盘格 solvePnP 失败: {exc}")
            return
        if not ok or float(tvec[2]) <= 0.0:
            self._capture_failed("棋盘格位于相机后方或 solvePnP 无效")
            return
        rotation, _ = cv2.Rodrigues(rvec)
        camera_from_board = np.eye(4, dtype=float)
        camera_from_board[:3, :3] = rotation
        camera_from_board[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
        projected, _ = cv2.projectPoints(
            self._board_points,
            rvec,
            tvec,
            self._camera_matrix,
            self._distortion,
        )
        error_px = float(
            np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1)))
        )
        if error_px > float(self.get_parameter("max_single_view_error_px").value):
            self._capture_failed(f"单帧棋盘格重投影误差过大: {error_px:.2f}px")
            return

        self._capture_image = image
        self._capture_points = corners
        self._capture_camera_from_board = camera_from_board
        self._capture_reprojection_error = error_px
        if not self._pose_client.service_is_ready():
            self._capture_failed("等待 /kinematics/get_current_pose 服务")
            return
        self._pending_future = self._pose_client.call_async(GetRobotPose.Request())
        self._pending_kind = "pose"
        self._auto_state = "capture_wait"

    def _finish_pose_future(self, future: Any) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self._capture_failed(f"获取机械臂位姿失败: {exc}")
            return
        if not result.success or not result.solution:
            self._capture_failed("获取机械臂位姿返回无效结果")
            return

        pose = result.pose
        world_from_ee = pose_to_matrix(
            [pose.position.x, pose.position.y, pose.position.z],
            [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
        )
        if self._is_duplicate_pose(world_from_ee):
            self._capture_failed("机械臂姿态与已有样本过近，当前样本丢弃")
            return
        if self._capture_camera_from_board is None or self._capture_points is None:
            self._capture_failed("内部采集缓存为空")
            return

        image_path = ""
        if self._capture_image is not None:
            try:
                os.makedirs(self._sample_dir, exist_ok=True)
                image_path = os.path.join(self._sample_dir, f"sample_{len(self._samples) + 1:03d}.png")
                cv2.imwrite(image_path, self._capture_image)
            except Exception as exc:
                self.get_logger().warning(f"样本图像保存失败，继续标定: {exc}")

        self._samples.append(
            CalibrationSample(
                world_from_ee=world_from_ee,
                camera_from_board=self._capture_camera_from_board.copy(),
                image_points=self._capture_points.copy(),
                reprojection_error_px=self._capture_reprojection_error,
                image_path=image_path,
            )
        )
        self.get_logger().info(
            f"已采集有效样本 {len(self._samples)}，单帧重投影误差 "
            f"{self._capture_reprojection_error:.2f}px"
        )
        self._clear_capture_cache()

        if self._auto_active:
            self._auto_index += 1
            if (
                len(self._samples) >= int(self.get_parameter("target_samples").value)
                or self._auto_index >= len(self._auto_poses)
            ):
                self._finish_auto()
            else:
                self._auto_state = "move_request"
        else:
            self._auto_state = "idle"
            self._publish_status("手动样本已保存，可继续 ~/capture 或调用 ~/solve")

    def _is_duplicate_pose(self, candidate: np.ndarray) -> bool:
        translation_limit = float(self.get_parameter("duplicate_translation_m").value)
        rotation_limit = math.radians(float(self.get_parameter("duplicate_rotation_deg").value))
        for sample in self._samples:
            relative = np.linalg.inv(sample.world_from_ee) @ candidate
            cosine = np.clip((np.trace(relative[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
            rotation = math.acos(float(cosine))
            translation = float(np.linalg.norm(relative[:3, 3]))
            if translation < translation_limit and rotation < rotation_limit:
                return True
        return False

    def _capture_failed(self, message: str) -> None:
        self.get_logger().warning(message)
        self._clear_capture_cache()
        if self._auto_active:
            self._auto_index += 1
            if self._auto_index >= len(self._auto_poses):
                self._finish_auto()
            else:
                self._auto_state = "move_request"
        else:
            self._auto_state = "idle"
            self._manual_capture_pending = False

    def _finish_auto(self) -> None:
        self._auto_active = False
        self._auto_state = "idle"
        if len(self._samples) < int(self.get_parameter("minimum_samples").value):
            message = (
                f"自动采集结束，但有效样本只有 {len(self._samples)}，"
                f"至少需要 {int(self.get_parameter('minimum_samples').value)} 个"
            )
            self._publish_status(message)
            self.get_logger().error(message)
            return
        success, message = self._solve_and_write()
        self._publish_status(message)
        if success:
            self.get_logger().info(message)
        else:
            self.get_logger().error(message)

    def _solve_and_write(self) -> tuple[bool, str]:
        minimum = int(self.get_parameter("minimum_samples").value)
        if len(self._samples) < minimum:
            return False, f"有效样本不足: {len(self._samples)}/{minimum}"
        if self._camera_matrix is None:
            return False, "尚未收到 camera_info"
        try:
            result = solve_hand_eye(
                self._samples,
                self._camera_matrix,
                self._distortion,
                self._board_points,
                method=self._method,
            )
            data = self._result_yaml(result)
            self._write_yaml(data)
        except Exception as exc:
            self.get_logger().error(f"手眼标定求解异常: {exc}")
            return False, f"手眼标定求解失败: {exc}"

        quality = result["quality"]
        warnings: list[str] = []
        if quality["reprojection_rms_px"] > 2.0:
            warnings.append(f"重投影 RMS={quality['reprojection_rms_px']:.2f}px")
        if quality["board_translation_rms_m"] > 0.01:
            warnings.append(f"固定板平移 RMS={quality['board_translation_rms_m']:.4f}m")
        if quality["board_rotation_rms_deg"] > 2.0:
            warnings.append(f"固定板旋转 RMS={quality['board_rotation_rms_deg']:.2f}deg")
        if quality["max_pairwise_rotation_deg"] < 15.0:
            warnings.append("姿态旋转激励不足，建议增加 pitch/yaw 变化")
        suffix = "；质量警告: " + ", ".join(warnings) if warnings else ""
        return True, f"标定完成，结果已写入 {self._output_file}{suffix}"

    def _result_yaml(self, result: dict[str, Any]) -> dict[str, Any]:
        hand_eye = result["T_ee_camera"]
        samples = []
        for index, sample in enumerate(self._samples, start=1):
            samples.append(
                {
                    "index": index,
                    "image_path": sample.image_path,
                    "single_view_reprojection_error_px": sample.reprojection_error_px,
                    "T_world_ee": matrix_to_list(sample.world_from_ee),
                    "T_camera_board": matrix_to_list(sample.camera_from_board),
                }
            )
        return {
            "schema_version": 1,
            "calibration_type": "eye_in_hand",
            "description": "A3 chessboard, 9x13 complete squares, 30 mm square size",
            "board": {
                "paper": "A3",
                "squares_x": int(self.get_parameter("board_squares_x").value),
                "squares_y": int(self.get_parameter("board_squares_y").value),
                "inner_corners_x": self._pattern_size[0],
                "inner_corners_y": self._pattern_size[1],
                "square_size_m": float(self.get_parameter("square_size_m").value),
            },
            "frames": {
                "base_frame": "world",
                "ee_frame": "kinematics_get_current_pose",
                "camera_frame": "rgb_optical_frame",
                "board_frame": "chessboard",
                "convention": "T_world_ee * T_ee_camera * T_camera_board",
            },
            "camera": {
                "image_topic": str(self.get_parameter("image_topic").value),
                "camera_info_topic": str(self.get_parameter("camera_info_topic").value),
                "camera_matrix": self._camera_matrix.tolist() if self._camera_matrix is not None else [],
                "distortion": self._distortion.reshape(-1).tolist(),
            },
            "hand_eye": {
                "method": self._method_name,
                "T_ee_camera": matrix_to_list(result["T_ee_camera"]),
                "T_camera_ee": matrix_to_list(result["T_camera_ee"]),
                "T_world_board": matrix_to_list(result["T_world_board"]),
            },
            "ee_from_camera": flatten_matrix(hand_eye),
            "quality": result["quality"],
            "samples": samples,
            "legacy_transform_yaml": {
                "keep_existing_keys": [
                    "corners",
                    "extristric",
                    "plane",
                    "white_area_pose_cam",
                    "white_area_pose_world",
                ],
                "copy_key": "ee_from_camera",
                "warning": "不要把 ee_from_camera 直接改写成 extristric；两者含义不同。",
            },
        }

    def _write_yaml(self, data: dict[str, Any]) -> None:
        output = Path(self._output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            shutil.copy2(output, str(output) + ".bak")
        fd, temporary = tempfile.mkstemp(prefix="hand_eye_", suffix=".yaml", dir=output.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _clear_capture_cache(self) -> None:
        self._capture_image = None
        self._capture_points = None
        self._capture_camera_from_board = None
        self._capture_reprojection_error = 0.0

    def _publish_status(self, text: str) -> None:
        message = String()
        message.data = text
        self._status_pub.publish(message)


def main() -> None:
    rclpy.init()
    node = HandEyeCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
