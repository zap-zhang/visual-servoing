<!-- 文件作用：说明混合视觉伺服包的运行流程、检测器选择、采摘阶段、启动方法和坐标约定。 -->
# 这是ai写的代码，不要拿去用
# JetArm Hybrid Visual Servoing

This package provides an extensible IBVS/PBVS controller for the JetArm ROS2 stack.
It uses the existing `/kinematics/set_pose_target` service and `/servo_controller`
topic because the current arm driver exposes Cartesian target-to-IK motion rather
than a Cartesian velocity interface.

## Runtime flow

```text
RGB image -> detector adapter -> Detection
Depth image + CameraInfo --------------------+
                                            |
Current end-effector pose <- FK service <---+-- control core
                                            |
                              PBVS / IBVS / Hybrid
                                            |
                         IK service -> servo pulse topic
```

The `Detection` protocol contains only class name, confidence, bounding box,
centroid, and timestamp. The current adapter can consume the existing
`interfaces/ObjectsInfo` topic or load an Ultralytics model directly. A trained
YOLOv11 strawberry model is therefore configured by changing `model_path` and,
when needed, `target_class`; the control code does not import a YOLO result type.

## Mode and switch distance

Set these parameters in `config/hybrid_servo.yaml`:

```yaml
servo_mode: hybrid
hybrid_switch_distance_m: 0.08
```

The hybrid controller uses PBVS farther than the switch distance and increases
the IBVS contribution continuously as the end effector approaches the target.
Set `servo_mode` to `pbvs` or `ibvs` for isolated testing. The config is installed
with the package and can be replaced at launch with `config:=...`.

## Detector choices

`detector_backend: topic` consumes `interfaces/ObjectsInfo`. The default
`detector_box_format: auto` treats four values as `xyxy` and five or more values
as `xywhr`, matching the current OBB YOLO node. Set it explicitly to `xyxy` for a
normal YOLO detector.

`detector_backend: ultralytics` subscribes to the RGB topic and runs the configured
Ultralytics model in the node. This is convenient for development; deployment can
keep inference in a separate node and use the topic adapter for lower coupling.

## Start

```bash
ros2 launch visual_servoing hybrid_servo.launch.py
ros2 service call /hybrid_visual_servo/start std_srvs/srv/Trigger {}
```

The node defaults to `enabled: false` so a launch does not move the arm before
the camera, detector, calibration, and IK services are confirmed. The workspace,
confidence, target timeout, speed, and per-cycle step are all configurable safety
limits.

## Strawberry pick sequence

The start service runs the following deterministic sequence:

```text
ACQUIRE -> PREGRASP -> APPROACH -> GRASP -> LIFT -> SUCCEEDED
```

The task publishes phase changes on `~/status` as `std_msgs/String`. The target
point is filtered and jump-limited before it is used. `PREGRASP`, `APPROACH`,
and `LIFT` offsets are configured in world coordinates. The gripper uses servo
ID `10` by default, with `gripper_open_pulse` and `gripper_close_pulse` as
separate parameters. After closing, a short visual dropout does not cancel the
lift because the state machine retains the last accepted target.

The current arm driver has no Cartesian velocity command. Each control cycle
therefore computes a bounded Cartesian position step, calls
`/kinematics/set_pose_target`, and publishes the returned five arm pulses to
`/servo_controller`. Validate the hand-eye transform and workspace limits in
simulation before enabling the start service on hardware.

## Calibration and coordinate conventions

`ee_from_camera` is a flattened 16-value `T_ee_camera`, using the same calibrated hand-eye convention
already used by the existing `app/calibration.py`. `GetRobotPose` supplies
`T_world_ee`. Depth is converted with the RGB `CameraInfo` pinhole matrix and the
resulting point is transformed as:

```text
P_world = T_world_ee * T_ee_camera * P_camera
```

Verify this convention with a known point before enabling motion on hardware.

## Chessboard hand-eye calibration

The package includes an eye-in-hand calibration node for an A3 chessboard
printed at 100% scale. Use 9 x 13 complete squares, each 30 mm, which gives
OpenCV an 8 x 12 inner-corner pattern. Mount the board rigidly in front of the
arm, keep its printed orientation fixed, and make sure the entire pattern is
visible in every accepted image.

Start the camera, arm driver, and kinematics service first, then run:

```bash
ros2 launch visual_servoing hand_eye_calibration.launch.py
```

The node provides `capture`, `solve`, `start_auto`, `stop`, and `reset` Trigger
services under `/hand_eye_calibration`. For manual collection, move the arm to
a distinct pose, wait until it stops, and call `capture`; collect at least 12
valid samples, preferably 20 or more. Change both position and orientation.
Translations alone or repeated images at one pose do not constrain hand-eye
calibration reliably.

`start_auto` executes the editable `auto_pose_list` through the existing
position-plus-pitch IK service. The example coordinates are not validated for
your board placement or hardware. Check every target in simulation and confirm
the board remains visible before using automatic motion on the real arm. The
driver cannot command a fully specified 6D pose, so manual capture is the
fallback when its IK-selected orientations do not provide enough rotation
diversity.

The node writes a separate `hand_eye_result.yaml` and sample images. It does
not edit the legacy `transform.yaml`. Copy `ee_from_camera` into the visual
servoing configuration after checking the reported quality values. Keep legacy
`corners`, `extristric`, `plane`, and `white_area_pose_world` unchanged; those
keys describe the older fixed-work-plane pipeline, not the hand-eye transform.

The board-to-camera poses from a symmetric chessboard can have a 180-degree
ordering ambiguity. Keep the board upright and its printed orientation
unchanged; reject a sample if detected corner order visibly flips. For larger
orientation changes, an asymmetric ChArUco or AprilGrid target is more robust.
