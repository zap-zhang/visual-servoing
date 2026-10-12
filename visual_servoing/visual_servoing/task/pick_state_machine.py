# 文件作用：实现草莓采摘状态机，管理目标获取、预抓取、接近、闭合夹爪和抬升阶段。
# 对目标位置进行滤波及跳变过滤，输出阶段目标、夹爪动作和超时失败原因，不直接控制硬件。
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PickPhase(str, Enum):
    IDLE = "idle"
    ACQUIRE = "acquire"
    PREGRASP = "pregrasp"
    APPROACH = "approach"
    GRASP = "grasp"
    LIFT = "lift"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class GripperAction(str, Enum):
    HOLD = "hold"
    OPEN = "open"
    CLOSE = "close"


@dataclass(slots=True, frozen=True)
class PickConfig:
    pregrasp_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.05)
    grasp_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.015)
    lift_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.10)
    position_tolerance_m: float = 0.012
    pregrasp_tolerance_m: float = 0.018
    target_timeout_s: float = 0.50
    acquisition_timeout_s: float = 20.0
    approach_timeout_s: float = 20.0
    grasp_settle_s: float = 0.8
    lift_timeout_s: float = 8.0
    max_target_jump_m: float = 0.06
    target_filter_alpha: float = 0.35

    def __post_init__(self) -> None:
        for name in (
            "position_tolerance_m",
            "pregrasp_tolerance_m",
            "target_timeout_s",
            "acquisition_timeout_s",
            "approach_timeout_s",
            "grasp_settle_s",
            "lift_timeout_s",
            "max_target_jump_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        alpha = float(self.target_filter_alpha)
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("target_filter_alpha must be in (0, 1]")
        for name in (
            "pregrasp_offset_world",
            "grasp_offset_world",
            "lift_offset_world",
        ):
            vector = np.asarray(getattr(self, name), dtype=float).reshape(3)
            if not np.isfinite(vector).all():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, tuple(float(x) for x in vector))


@dataclass(slots=True, frozen=True)
class PickCommand:
    phase: PickPhase
    goal_position_world: np.ndarray | None
    gripper: GripperAction
    hold_position: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        if self.goal_position_world is not None:
            goal = np.asarray(self.goal_position_world, dtype=float).reshape(3)
            if not np.isfinite(goal).all():
                raise ValueError("goal_position_world must be finite")
            object.__setattr__(self, "goal_position_world", goal)


class PickStateMachine:
    """Deterministic strawberry pick sequence driven by visual target updates."""

    def __init__(self, config: PickConfig):
        self.config = config
        self.phase = PickPhase.IDLE
        self.phase_started_s = 0.0
        self._filtered_target: np.ndarray | None = None
        self._target_last_seen_s = -np.inf
        self.failure_reason = ""

    @property
    def filtered_target(self) -> np.ndarray | None:
        return None if self._filtered_target is None else self._filtered_target.copy()

    def start(self, now_s: float) -> None:
        self.phase = PickPhase.ACQUIRE
        self.phase_started_s = float(now_s)
        self._filtered_target = None
        self._target_last_seen_s = -np.inf
        self.failure_reason = ""

    def stop(self, reason: str = "stopped") -> None:
        self.phase = PickPhase.IDLE
        self.failure_reason = reason

    def update_target(self, target_position_world: np.ndarray, now_s: float) -> bool:
        target = np.asarray(target_position_world, dtype=float).reshape(3)
        if not np.isfinite(target).all():
            return False
        if (
            self._filtered_target is not None
            and np.linalg.norm(target - self._filtered_target)
            > self.config.max_target_jump_m
        ):
            return False
        if self._filtered_target is None:
            self._filtered_target = target.copy()
        else:
            alpha = self.config.target_filter_alpha
            self._filtered_target = (
                alpha * target + (1.0 - alpha) * self._filtered_target
            )
        self._target_last_seen_s = float(now_s)
        return True

    def step(self, now_s: float, ee_position_world: np.ndarray) -> PickCommand:
        now = float(now_s)
        ee = np.asarray(ee_position_world, dtype=float).reshape(3)
        if not np.isfinite(ee).all():
            return self._fail("invalid end-effector position")
        if self.phase in {PickPhase.IDLE, PickPhase.SUCCEEDED, PickPhase.FAILED}:
            return PickCommand(self.phase, None, GripperAction.OPEN, True, self.failure_reason)

        target = self._filtered_target
        target_valid = target is not None and (
            now - self._target_last_seen_s <= self.config.target_timeout_s
        )
        if self.phase is PickPhase.ACQUIRE:
            if not target_valid:
                if now - self.phase_started_s > self.config.acquisition_timeout_s:
                    return self._fail("strawberry acquisition timed out")
                return PickCommand(self.phase, ee.copy(), GripperAction.OPEN, True, "waiting for strawberry")
            self.phase = PickPhase.PREGRASP
            self.phase_started_s = now

        if self.phase is PickPhase.PREGRASP:
            if not target_valid:
                return self._fail("strawberry lost before pregrasp")
            goal = target + np.asarray(self.config.pregrasp_offset_world)
            if np.linalg.norm(goal - ee) <= self.config.pregrasp_tolerance_m:
                self.phase = PickPhase.APPROACH
                self.phase_started_s = now
            else:
                return PickCommand(self.phase, goal, GripperAction.OPEN)

        if self.phase is PickPhase.APPROACH:
            if not target_valid:
                return self._fail("strawberry lost during approach")
            goal = target + np.asarray(self.config.grasp_offset_world)
            if np.linalg.norm(goal - ee) <= self.config.position_tolerance_m:
                self.phase = PickPhase.GRASP
                self.phase_started_s = now
                return PickCommand(self.phase, ee.copy(), GripperAction.CLOSE, True, "close gripper")
            if now - self.phase_started_s > self.config.approach_timeout_s:
                return self._fail("approach timed out")
            return PickCommand(self.phase, goal, GripperAction.OPEN)

        if self.phase is PickPhase.GRASP:
            if now - self.phase_started_s >= self.config.grasp_settle_s:
                self.phase = PickPhase.LIFT
                self.phase_started_s = now
            else:
                return PickCommand(self.phase, ee.copy(), GripperAction.CLOSE, True, "settling grasp")

        if self.phase is PickPhase.LIFT:
            goal = (
                target + np.asarray(self.config.grasp_offset_world)
                + np.asarray(self.config.lift_offset_world)
                if target is not None
                else ee + np.asarray(self.config.lift_offset_world)
            )
            if np.linalg.norm(goal - ee) <= self.config.position_tolerance_m:
                self.phase = PickPhase.SUCCEEDED
                return PickCommand(self.phase, ee.copy(), GripperAction.CLOSE, True, "strawberry lifted")
            if now - self.phase_started_s > self.config.lift_timeout_s:
                return self._fail("lift timed out")
            return PickCommand(self.phase, goal, GripperAction.CLOSE)

        return self._fail("unhandled pick phase")

    def _fail(self, reason: str) -> PickCommand:
        self.phase = PickPhase.FAILED
        self.failure_reason = reason
        return PickCommand(self.phase, None, GripperAction.OPEN, True, reason)
