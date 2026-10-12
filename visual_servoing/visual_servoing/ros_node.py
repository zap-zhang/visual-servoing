# 文件作用：混合视觉伺服的 ROS2 主节点，连接视觉感知、采摘状态机和机械臂控制接口。
# 接收相机与检测数据，估计草莓位置，通过 IK 服务和舵机话题执行运动及夹爪动作。
from __future__ import annotations

import time
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from interfaces.msg import ObjectsInfo
from kinematics_msgs.srv import GetRobotPose, SetRobotPose
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .core.controller import HybridVisualServo, ServoConfig
from .core.geometry import camera_point_to_world, depth_to_camera_point, pose_to_matrix
from .core.types import Detection, FeatureObservation, ServoMode
from .perception.depth import DepthEstimator
from .perception.detectors import UltralyticsDetector
from .task.pick_state_machine import (
    GripperAction,
    PickCommand,
    PickConfig,
    PickPhase,
    PickStateMachine,
)


class HybridVisualServoNode(Node):
    """ROS adapter for the detector-neutral hybrid visual servo controller."""

    def __init__(self) -> None:
        super().__init__("hybrid_visual_servo")
        self._declare_parameters()
        self.enabled = bool(self.get_parameter("enabled").value)
        self._bridge = CvBridge()
        self._latest_detection: Detection | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_depth_time_s = 0.0
        self._intrinsics: np.ndarray | None = None
        self._ee_pose: Pose | None = None
        self._pose_future = None
        self._ik_future = None
        self._last_command_time_s = 0.0
        self._last_warning_time_s = 0.0
        self._last_gripper_pulse: int | None = None
        self._last_task_phase = PickPhase.IDLE
        self._motion_generation = 0

        self._ee_from_camera = np.asarray(
            self.get_parameter("ee_from_camera").value, dtype=float
        ).reshape(4, 4)
        self._depth_estimator = DepthEstimator(
            float(self.get_parameter("depth_scale_m_per_unit").value)
        )
        self._controller = HybridVisualServo(self._load_servo_config())
        self._pick_task = PickStateMachine(self._load_pick_config())
        self._detector = None
        if self.get_parameter("detector_backend").value == "ultralytics":
            self._detector = UltralyticsDetector(
                model_path=str(self.get_parameter("model_path").value),
                target_class=str(self.get_parameter("target_class").value),
                confidence_threshold=float(
                    self.get_parameter("detector_confidence").value
                ),
                image_size=int(self.get_parameter("detector_image_size").value),
                device=str(self.get_parameter("detector_device").value),
            )

        self._servo_pub = self.create_publisher(
            ServosPosition,
            "/servo_controller",
            1,
        )
        self._status_pub = self.create_publisher(String, "~/status", 1)
        self._pose_client = self.create_client(
            GetRobotPose, "/kinematics/get_current_pose"
        )
        self._ik_client = self.create_client(
            SetRobotPose, "/kinematics/set_pose_target"
        )

        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            1,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._depth_callback,
            1,
        )
        if self._detector is None:
            self.create_subscription(
                ObjectsInfo,
                str(self.get_parameter("detector_topic").value),
                self._detection_callback,
                1,
            )
        else:
            self.create_subscription(
                Image,
                str(self.get_parameter("rgb_topic").value),
                self._rgb_callback,
                1,
            )

        self.create_service(Trigger, "~/start", self._start_callback)
        self.create_service(Trigger, "~/stop", self._stop_callback)
        self.create_service(Trigger, "~/reset", self._reset_callback)
        period = 1.0 / float(self.get_parameter("control_rate_hz").value)
        self.create_timer(period, self._control_timer_callback)
        self.get_logger().info(
            f"hybrid strawberry picker ready: mode={self._controller.config.mode.value} "
            f"switch_distance={self._controller.config.hybrid_switch_distance_m:.3f}m "
            f"backend={self.get_parameter('detector_backend').value}",
        )

    def _declare_parameters(self) -> None:
        parameters: dict[str, Any] = {
            "enabled": False,
            "detector_backend": "topic",
            "detector_topic": "/yolov8/object_detect",
            "detector_box_format": "auto",
            "rgb_topic": "/depth_cam/rgb/image_raw",
            "depth_topic": "/depth_cam/depth/image_raw",
            "camera_info_topic": "/depth_cam/rgb/camera_info",
            "model_path": "strawberry_yolov11.pt",
            "target_class": "strawberry",
            "detector_confidence": 0.50,
            "detector_image_size": 640,
            "detector_device": "",
            "depth_scale_m_per_unit": 0.001,
            "depth_timeout_s": 0.50,
            "camera_role": "eye_in_hand",
            "ee_from_camera": [
                0.0,
                0.0,
                1.0,
                -0.101,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.045,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            "servo_mode": "hybrid",
            "pbvs_gain": 1.2,
            "ibvs_gain": 0.8,
            "depth_gain": 0.6,
            "desired_depth_m": 0.18,
            "max_speed_mps": 0.04,
            "control_period_s": 0.10,
            "hybrid_switch_distance_m": 0.08,
            "ibvs_target_pixel_mode": "center",
            "ibvs_target_pixel": [320.0, 240.0],
            "max_step_m": 0.006,
            "target_timeout_s": 0.50,
            "minimum_confidence": 0.50,
            "pitch_deg": -90.0,
            "pitch_range_deg": [-100.0, -80.0],
            "ik_resolution_deg": 1.0,
            "ik_duration_s": 0.10,
            "arm_command_duration_s": 0.10,
            "control_rate_hz": 10.0,
            "workspace_min_m": [0.05, -0.25, 0.02],
            "workspace_max_m": [0.35, 0.25, 0.35],
            "gripper_open_pulse": 200,
            "gripper_close_pulse": 600,
            "gripper_id": 10,
            "pick_pregrasp_offset_world": [0.0, 0.0, 0.05],
            "pick_grasp_offset_world": [0.0, 0.0, 0.015],
            "pick_lift_offset_world": [0.0, 0.0, 0.10],
            "pick_position_tolerance_m": 0.012,
            "pick_pregrasp_tolerance_m": 0.018,
            "pick_acquisition_timeout_s": 20.0,
            "pick_approach_timeout_s": 20.0,
            "pick_grasp_settle_s": 0.8,
            "pick_lift_timeout_s": 8.0,
            "pick_max_target_jump_m": 0.06,
            "pick_target_filter_alpha": 0.35,
        }
        for name, default in parameters.items():
            self.declare_parameter(name, default)

    def _load_servo_config(self) -> ServoConfig:
        target_pixel = None
        if self.get_parameter("ibvs_target_pixel_mode").value == "fixed":
            target_pixel = tuple(
                float(x) for x in self.get_parameter("ibvs_target_pixel").value
            )
        return ServoConfig(
            mode=ServoMode(str(self.get_parameter("servo_mode").value)),
            pbvs_gain=float(self.get_parameter("pbvs_gain").value),
            ibvs_gain=float(self.get_parameter("ibvs_gain").value),
            depth_gain=float(self.get_parameter("depth_gain").value),
            max_speed_mps=float(self.get_parameter("max_speed_mps").value),
            control_period_s=float(self.get_parameter("control_period_s").value),
            hybrid_switch_distance_m=float(
                self.get_parameter("hybrid_switch_distance_m").value
            ),
            ibvs_target_pixel=target_pixel,
            max_step_m=float(self.get_parameter("max_step_m").value),
        )

    def _load_pick_config(self) -> PickConfig:
        return PickConfig(
            pregrasp_offset_world=tuple(
                float(x)
                for x in self.get_parameter("pick_pregrasp_offset_world").value
            ),
            grasp_offset_world=tuple(
                float(x)
                for x in self.get_parameter("pick_grasp_offset_world").value
            ),
            lift_offset_world=tuple(
                float(x)
                for x in self.get_parameter("pick_lift_offset_world").value
            ),
            position_tolerance_m=float(
                self.get_parameter("pick_position_tolerance_m").value
            ),
            pregrasp_tolerance_m=float(
                self.get_parameter("pick_pregrasp_tolerance_m").value
            ),
            target_timeout_s=float(self.get_parameter("target_timeout_s").value),
            acquisition_timeout_s=float(
                self.get_parameter("pick_acquisition_timeout_s").value
            ),
            approach_timeout_s=float(
                self.get_parameter("pick_approach_timeout_s").value
            ),
            grasp_settle_s=float(self.get_parameter("pick_grasp_settle_s").value),
            lift_timeout_s=float(self.get_parameter("pick_lift_timeout_s").value),
            max_target_jump_m=float(
                self.get_parameter("pick_max_target_jump_m").value
            ),
            target_filter_alpha=float(
                self.get_parameter("pick_target_filter_alpha").value
            ),
        )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._intrinsics = np.asarray(message.k, dtype=float).reshape(3, 3)

    def _depth_callback(self, message: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
            self._latest_depth = np.asarray(depth)
            self._latest_depth_time_s = self._now_s()
        except Exception as exc:  # pragma: no cover - depends on ROS encoding support
            self.get_logger().warning("depth conversion failed: %s", exc)

    def _rgb_callback(self, message: Image) -> None:
        if self._detector is None:
            return
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            detection = self._detector.detect(np.asarray(image), self._now_s())
            if detection is not None:
                self._latest_detection = detection
        except Exception as exc:  # pragma: no cover - requires optional model runtime
            self._warn_throttled("detector failed: %s", exc)

    def _detection_callback(self, message: ObjectsInfo) -> None:
        target_class = str(self.get_parameter("target_class").value)
        minimum_confidence = float(self.get_parameter("minimum_confidence").value)
        candidates = [
            item
            for item in message.objects
            if (not target_class or item.class_name == target_class)
            and float(item.score) >= minimum_confidence
            and len(item.box) >= 4
        ]
        if not candidates:
            return
        item = max(candidates, key=lambda value: float(value.score))
        box = np.asarray(item.box, dtype=float)
        box_format = str(self.get_parameter("detector_box_format").value).lower()
        if box_format == "auto":
            box_format = "xywhr" if len(box) >= 5 else "xyxy"
        if box_format in {"xywh", "xywhr"}:
            center_x, center_y, width, height = box[:4]
            bbox = np.array(
                [
                    center_x - width / 2.0,
                    center_y - height / 2.0,
                    center_x + width / 2.0,
                    center_y + height / 2.0,
                ]
            )
        else:
            bbox = box[:4]
        self._latest_detection = Detection(
            class_name=str(item.class_name),
            confidence=float(item.score),
            bbox_xyxy=bbox,
            centroid_px=np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]),
            stamp_s=self._now_s(),
        )

    def _start_callback(self, request: Trigger.Request, response: Trigger.Response):
        del request
        self._motion_generation += 1
        self._pose_future = None
        self._ik_future = None
        self.enabled = True
        self._pick_task.start(self._now_s())
        self._last_task_phase = PickPhase.IDLE
        self._publish_task_status(PickPhase.ACQUIRE)
        self._publish_gripper_action(GripperAction.OPEN, force=True)
        response.success = True
        response.message = "strawberry pick task started"
        return response

    def _stop_callback(self, request: Trigger.Request, response: Trigger.Response):
        del request
        self._motion_generation += 1
        self.enabled = False
        self._pick_task.stop("stopped by service request")
        self._latest_detection = None
        self._pose_future = None
        self._ik_future = None
        self._publish_gripper_action(GripperAction.OPEN, force=True)
        self._publish_task_status(PickPhase.IDLE, "stopped")
        response.success = True
        response.message = "strawberry pick task stopped"
        return response

    def _reset_callback(self, request: Trigger.Request, response: Trigger.Response):
        del request
        self._motion_generation += 1
        self.enabled = False
        self._pick_task.stop("reset")
        self._latest_detection = None
        self._ee_pose = None
        self._pose_future = None
        self._ik_future = None
        self._publish_gripper_action(GripperAction.OPEN, force=True)
        self._publish_task_status(PickPhase.IDLE, "reset")
        response.success = True
        response.message = "strawberry pick task reset"
        return response

    def _control_timer_callback(self) -> None:
        if not self.enabled or self._ik_future is not None:
            return
        if self._pick_task.phase in {
            PickPhase.IDLE,
            PickPhase.SUCCEEDED,
            PickPhase.FAILED,
        }:
            return
        if self._pose_future is not None:
            return
        if not self._pose_client.service_is_ready():
            self._warn_throttled("waiting for /kinematics/get_current_pose")
            return
        generation = self._motion_generation
        self._pose_future = self._pose_client.call_async(GetRobotPose.Request())
        self._pose_future.add_done_callback(
            lambda future: self._pose_done_callback(future, generation)
        )

    def _pose_done_callback(self, future, generation: int) -> None:
        if generation != self._motion_generation or not self.enabled:
            if self._pose_future is future:
                self._pose_future = None
            return
        self._pose_future = None
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - ROS service failure
            self._warn_throttled("current pose request failed: %s", exc)
            return
        if response is None or not response.success or not response.solution:
            self._warn_throttled("current pose is unavailable")
            return
        self._ee_pose = response.pose
        self._compute_and_request_motion()

    def _compute_and_request_motion(self) -> None:
        if self._ee_pose is None:
            return
        ee_position = np.array(
            [
                self._ee_pose.position.x,
                self._ee_pose.position.y,
                self._ee_pose.position.z,
            ],
            dtype=float,
        )
        world_from_ee = pose_to_matrix(
            ee_position,
            [
                self._ee_pose.orientation.w,
                self._ee_pose.orientation.x,
                self._ee_pose.orientation.y,
                self._ee_pose.orientation.z,
            ],
        )

        visual_target = self._estimate_target_world(world_from_ee)
        if visual_target is not None:
            target_world, feature = visual_target
            if not self._pick_task.update_target(target_world, self._now_s()):
                self._warn_throttled("rejected implausible strawberry target jump")
        else:
            feature = None

        pick_command = self._pick_task.step(self._now_s(), ee_position)
        self._publish_pick_status(pick_command)
        self._publish_gripper_action(pick_command.gripper)

        if pick_command.goal_position_world is None or pick_command.hold_position:
            return

        goal_position = self._clip_workspace(pick_command.goal_position_world)
        use_visual_servo = pick_command.phase in {
            PickPhase.PREGRASP,
            PickPhase.APPROACH,
        }
        if use_visual_servo and feature is not None and self._intrinsics is not None:
            camera_rotation_world = world_from_ee[:3, :3] @ self._ee_from_camera[:3, :3]
            try:
                servo_command = self._controller.compute(
                    ee_position_world=ee_position,
                    target_position_world=goal_position,
                    feature=feature,
                    camera_intrinsics=self._intrinsics,
                    camera_rotation_world=camera_rotation_world,
                    desired_depth_m=float(
                        self.get_parameter("desired_depth_m").value
                    ),
                )
                next_position = self._clip_workspace(
                    servo_command.next_position_world
                )
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                self._warn_throttled("visual servo computation failed: %s", exc)
                return
        else:
            # After a successful grasp, the stored target remains usable for a
            # bounded PBVS lift even when the strawberry leaves the camera view.
            next_position = self._bounded_position_step(ee_position, goal_position)

        if not self._ik_client.service_is_ready():
            self._warn_throttled("waiting for /kinematics/set_pose_target")
            return
        request = SetRobotPose.Request()
        request.position = [float(value) for value in next_position]
        request.pitch = float(self.get_parameter("pitch_deg").value)
        request.pitch_range = [
            float(value) for value in self.get_parameter("pitch_range_deg").value
        ]
        request.resolution = float(self.get_parameter("ik_resolution_deg").value)
        request.duration = float(self.get_parameter("ik_duration_s").value)
        generation = self._motion_generation
        self._ik_future = self._ik_client.call_async(request)
        self._ik_future.add_done_callback(
            lambda future: self._ik_done_callback(future, generation)
        )
        self._last_command_time_s = self._now_s()

    def _ik_done_callback(self, future, generation: int) -> None:
        if generation != self._motion_generation or not self.enabled:
            if self._ik_future is future:
                self._ik_future = None
            return
        self._ik_future = None
        try:
            response = future.result()
        except Exception as exc:  # pragma: no cover - ROS service failure
            self._warn_throttled("inverse kinematics request failed: %s", exc)
            return
        if response is None or not response.pulse or len(response.pulse) < 5:
            self._warn_throttled("target is unreachable")
            return
        pulse = [int(np.clip(value, 0, 1000)) for value in response.pulse[:5]]
        gripper_id = int(self.get_parameter("gripper_id").value)
        gripper_pulse = self._last_gripper_pulse
        if gripper_pulse is None:
            gripper_pulse = int(self.get_parameter("gripper_open_pulse").value)
        set_servo_position(
            self._servo_pub,
            float(self.get_parameter("arm_command_duration_s").value),
            tuple([(index + 1, value) for index, value in enumerate(pulse)]
                  + [(gripper_id, gripper_pulse)]),
        )

    def _estimate_target_world(
        self, world_from_ee: np.ndarray
    ) -> tuple[np.ndarray, FeatureObservation] | None:
        detection = self._latest_detection
        if (
            detection is None
            or self._latest_depth is None
            or self._intrinsics is None
            or not self._target_is_valid()
        ):
            return None
        depth_age = self._now_s() - self._latest_depth_time_s
        if depth_age > float(self.get_parameter("depth_timeout_s").value):
            self._warn_throttled("depth timeout; waiting for a fresh depth frame")
            return None
        depth_m = self._depth_estimator.estimate(
            self._latest_depth, detection.bbox_xyxy
        )
        if depth_m is None:
            self._warn_throttled("target depth is unavailable")
            return None
        camera_point = depth_to_camera_point(
            detection.centroid_px, depth_m, self._intrinsics
        )
        target_world = camera_point_to_world(
            camera_point, world_from_ee, self._ee_from_camera
        )
        return target_world, FeatureObservation(
            detection.centroid_px, depth_m, detection.confidence
        )

    def _bounded_position_step(
        self, current_position: np.ndarray, goal_position: np.ndarray
    ) -> np.ndarray:
        delta = np.asarray(goal_position, dtype=float) - np.asarray(
            current_position, dtype=float
        )
        step_limit = min(
            float(self.get_parameter("max_step_m").value),
            float(self.get_parameter("max_speed_mps").value)
            * float(self.get_parameter("control_period_s").value),
        )
        norm = float(np.linalg.norm(delta))
        if norm <= step_limit:
            return np.asarray(goal_position, dtype=float)
        return np.asarray(current_position, dtype=float) + delta / norm * step_limit

    def _publish_gripper_action(
        self, action: GripperAction, *, force: bool = False
    ) -> None:
        if action is GripperAction.HOLD and self._last_gripper_pulse is not None:
            return
        if action is GripperAction.CLOSE:
            pulse = int(self.get_parameter("gripper_close_pulse").value)
        else:
            pulse = int(self.get_parameter("gripper_open_pulse").value)
        if not force and self._last_gripper_pulse == pulse:
            return
        set_servo_position(
            self._servo_pub,
            float(self.get_parameter("arm_command_duration_s").value),
            ((int(self.get_parameter("gripper_id").value), pulse),),
        )
        self._last_gripper_pulse = pulse

    def _publish_pick_status(self, command: PickCommand) -> None:
        reason = command.reason if command.phase in {
            PickPhase.FAILED,
            PickPhase.SUCCEEDED,
        } else ""
        self._publish_task_status(command.phase, reason)

    def _publish_task_status(self, phase: PickPhase, reason: str = "") -> None:
        if phase == self._last_task_phase and phase not in {
            PickPhase.FAILED,
            PickPhase.SUCCEEDED,
        }:
            return
        message = f"phase={phase.value}"
        if reason:
            message += f" reason={reason}"
        self._status_pub.publish(String(data=message))
        self._last_task_phase = phase

    def _target_is_valid(self) -> bool:
        detection = self._latest_detection
        if detection is None:
            return False
        age = self._now_s() - detection.stamp_s
        timeout = float(self.get_parameter("target_timeout_s").value)
        if age > timeout:
            self._warn_throttled("target timeout; holding position")
            return False
        return detection.confidence >= float(
            self.get_parameter("minimum_confidence").value
        )

    def _clip_workspace(self, position: np.ndarray) -> np.ndarray:
        lower = np.asarray(self.get_parameter("workspace_min_m").value, dtype=float)
        upper = np.asarray(self.get_parameter("workspace_max_m").value, dtype=float)
        return np.clip(np.asarray(position, dtype=float), lower, upper)

    def _now_s(self) -> float:
        return time.monotonic()

    def _warn_throttled(self, message: str, *args: Any) -> None:
        now = self._now_s()
        if now - self._last_warning_time_s >= 2.0:
            self.get_logger().warning(message, *args)
            self._last_warning_time_s = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HybridVisualServoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
