# 文件作用：测试坐标转换、混合控制权重、单步位移限制和采摘状态机的基本流程。
# 测试不依赖 ROS 运行环境，不覆盖真实相机、ROS 通信或机械臂硬件动作。
import numpy as np
import pytest

from visual_servoing.core.controller import HybridVisualServo, ServoConfig
from visual_servoing.core.geometry import (
    camera_point_to_world,
    depth_to_camera_point,
)
from visual_servoing.core.types import FeatureObservation, ServoMode
from visual_servoing.task.pick_state_machine import (
    GripperAction,
    PickConfig,
    PickPhase,
    PickStateMachine,
)


def test_depth_projection_and_hand_eye_transform():
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    camera_point = depth_to_camera_point([60.0, 50.0], 0.2, intrinsics)
    world_from_ee = np.eye(4)
    world_from_ee[:3, 3] = [0.1, 0.0, 0.3]
    ee_from_camera = np.eye(4)
    ee_from_camera[:3, 3] = [0.02, 0.0, 0.0]
    result = camera_point_to_world(camera_point, world_from_ee, ee_from_camera)
    np.testing.assert_allclose(result, [0.14, 0.02, 0.5])


def test_hybrid_switch_distance_is_configurable():
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    feature = FeatureObservation([55.0, 40.0], 0.2)
    controller = HybridVisualServo(
        ServoConfig(
            mode=ServoMode.HYBRID,
            hybrid_switch_distance_m=0.10,
            max_speed_mps=1.0,
            control_period_s=0.1,
            max_step_m=1.0,
        )
    )
    command = controller.compute(
        ee_position_world=[0.0, 0.0, 0.0],
        target_position_world=[0.05, 0.0, 0.0],
        feature=feature,
        camera_intrinsics=intrinsics,
        camera_rotation_world=np.eye(3),
        desired_depth_m=0.2,
    )
    assert command.mode is ServoMode.HYBRID
    assert command.ibvs_weight == pytest.approx(0.5)


def test_controller_limits_one_step_motion():
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    command = HybridVisualServo(
        ServoConfig(mode=ServoMode.PBVS, max_speed_mps=1.0, control_period_s=1.0, max_step_m=0.01)
    ).compute(
        ee_position_world=[0.0, 0.0, 0.0],
        target_position_world=[1.0, 0.0, 0.0],
        feature=FeatureObservation([50.0, 40.0], 0.2),
        camera_intrinsics=intrinsics,
        camera_rotation_world=np.eye(3),
        desired_depth_m=0.2,
    )
    assert np.linalg.norm(command.next_position_world) == pytest.approx(0.01)


def test_pick_state_machine_runs_pregrasp_grasp_and_lift_sequence():
    task = PickStateMachine(
        PickConfig(
            pregrasp_offset_world=(0.0, 0.0, 0.05),
            grasp_offset_world=(0.0, 0.0, 0.01),
            lift_offset_world=(0.0, 0.0, 0.10),
            grasp_settle_s=0.5,
        )
    )
    target = np.array([0.20, 0.0, 0.10])
    task.start(0.0)
    task.update_target(target, 0.0)

    command = task.step(0.0, [0.0, 0.0, 0.0])
    assert command.phase is PickPhase.PREGRASP
    assert command.gripper is GripperAction.OPEN
    np.testing.assert_allclose(command.goal_position_world, target + [0.0, 0.0, 0.05])

    task.update_target(target, 1.0)
    command = task.step(1.0, target + [0.0, 0.0, 0.05])
    assert command.phase is PickPhase.APPROACH
    assert command.gripper is GripperAction.OPEN

    task.update_target(target, 2.0)
    command = task.step(2.0, target + [0.0, 0.0, 0.01])
    assert command.phase is PickPhase.GRASP
    assert command.gripper is GripperAction.CLOSE
    assert command.hold_position

    # The camera may lose the strawberry after the gripper closes.
    command = task.step(2.6, target + [0.0, 0.0, 0.01])
    assert command.phase is PickPhase.LIFT
    assert command.gripper is GripperAction.CLOSE
    np.testing.assert_allclose(
        command.goal_position_world, target + [0.0, 0.0, 0.11]
    )

    command = task.step(3.6, target + [0.0, 0.0, 0.11])
    assert command.phase is PickPhase.SUCCEEDED
    assert command.gripper is GripperAction.CLOSE
