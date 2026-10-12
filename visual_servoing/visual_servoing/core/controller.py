# 文件作用：实现独立于 ROS 的 PBVS、IBVS 和连续加权混合视觉伺服控制器。
# 根据位置与图像误差计算控制速度，输出经过限速、限步长处理的下一步末端位置。
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import clamp_norm, normalized_pixel, point_interaction_matrix
from .types import FeatureObservation, ServoCommand, ServoMode


@dataclass(slots=True, frozen=True)
class ServoConfig:
    mode: ServoMode = ServoMode.HYBRID
    pbvs_gain: float = 1.2
    ibvs_gain: float = 0.8
    depth_gain: float = 0.6
    max_speed_mps: float = 0.04
    control_period_s: float = 0.10
    hybrid_switch_distance_m: float = 0.08
    ibvs_target_pixel: tuple[float, float] | None = None
    max_step_m: float = 0.006

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", ServoMode(self.mode))
        for name in (
            "pbvs_gain",
            "ibvs_gain",
            "depth_gain",
            "max_speed_mps",
            "control_period_s",
            "hybrid_switch_distance_m",
            "max_step_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.ibvs_target_pixel is not None:
            target = np.asarray(self.ibvs_target_pixel, dtype=float).reshape(2)
            if not np.isfinite(target).all():
                raise ValueError("ibvs_target_pixel must be finite")
            object.__setattr__(self, "ibvs_target_pixel", (float(target[0]), float(target[1])))


class HybridVisualServo:
    """Compute a bounded Cartesian command independently of ROS or a detector."""

    def __init__(self, config: ServoConfig):
        self.config = config

    def compute(
        self,
        *,
        ee_position_world: np.ndarray,
        target_position_world: np.ndarray,
        feature: FeatureObservation,
        camera_intrinsics: np.ndarray,
        camera_rotation_world: np.ndarray,
        desired_depth_m: float,
    ) -> ServoCommand:
        ee = np.asarray(ee_position_world, dtype=float).reshape(3)
        target = np.asarray(target_position_world, dtype=float).reshape(3)
        rotation = np.asarray(camera_rotation_world, dtype=float).reshape(3, 3)
        if not np.isfinite(ee).all() or not np.isfinite(target).all():
            raise ValueError("world positions must be finite")
        if not np.isfinite(rotation).all():
            raise ValueError("camera rotation must be finite")

        pbvs_error = target - ee
        pbvs_velocity = clamp_norm(
            self.config.pbvs_gain * pbvs_error, self.config.max_speed_mps
        )
        ibvs_velocity, image_error_px = self._ibvs_velocity(
            feature=feature,
            camera_intrinsics=camera_intrinsics,
            camera_rotation_world=rotation,
            desired_depth_m=desired_depth_m,
        )

        distance = float(np.linalg.norm(pbvs_error))
        if self.config.mode is ServoMode.PBVS:
            mode = ServoMode.PBVS
            weight = 0.0
            velocity = pbvs_velocity
        elif self.config.mode is ServoMode.IBVS:
            mode = ServoMode.IBVS
            weight = 1.0
            velocity = ibvs_velocity
        else:
            mode = ServoMode.HYBRID
            weight = float(
                np.clip(
                    1.0 - distance / self.config.hybrid_switch_distance_m,
                    0.0,
                    1.0,
                )
            )
            velocity = (1.0 - weight) * pbvs_velocity + weight * ibvs_velocity
            velocity = clamp_norm(velocity, self.config.max_speed_mps)

        return ServoCommand(
            linear_velocity_world=velocity,
            next_position_world=ee + clamp_norm(
                velocity * self.config.control_period_s, self.config.max_step_m
            ),
            mode=mode,
            position_error_m=distance,
            image_error_px=image_error_px,
            ibvs_weight=weight,
        )

    def _ibvs_velocity(
        self,
        *,
        feature: FeatureObservation,
        camera_intrinsics: np.ndarray,
        camera_rotation_world: np.ndarray,
        desired_depth_m: float,
    ) -> tuple[np.ndarray, float]:
        if desired_depth_m <= 0.0 or not np.isfinite(desired_depth_m):
            raise ValueError("desired_depth_m must be positive and finite")
        target_pixel = self.config.ibvs_target_pixel
        if target_pixel is None:
            target_pixel = np.asarray(camera_intrinsics, dtype=float).reshape(3, 3)[
                :2, 2
            ]
        actual_xy = normalized_pixel(feature.pixel, camera_intrinsics)
        desired_xy = normalized_pixel(target_pixel, camera_intrinsics)
        image_error = actual_xy - desired_xy
        interaction = point_interaction_matrix(
            float(actual_xy[0]), float(actual_xy[1]), feature.depth_m
        )
        camera_velocity = -self.config.ibvs_gain * np.linalg.pinv(interaction) @ image_error
        camera_velocity[2] += self.config.depth_gain * (
            desired_depth_m - feature.depth_m
        )
        world_velocity = camera_rotation_world @ camera_velocity
        return clamp_norm(world_velocity, self.config.max_speed_mps), float(
            np.linalg.norm(np.asarray(feature.pixel) - np.asarray(target_pixel))
        )
