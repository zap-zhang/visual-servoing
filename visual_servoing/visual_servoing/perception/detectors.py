# 文件作用：定义统一检测器接口，并将 Ultralytics 模型的检测结果转换为 Detection。
# 按目标类别筛选并选择置信度最高的目标，支持通过模型路径替换 YOLOv8 或 YOLOv11 权重。
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..core.types import Detection


class Detector(Protocol):
    def detect(self, image_bgr: np.ndarray, stamp_s: float) -> Detection | None: ...


@dataclass(slots=True)
class UltralyticsDetector:
    """YOLO adapter. Changing YOLOv8 to YOLOv11 only changes model_path."""

    model_path: str
    target_class: str = "strawberry"
    confidence_threshold: float = 0.50
    image_size: int = 640
    device: str = ""
    _model: Any = None

    def __post_init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image_bgr: np.ndarray, stamp_s: float) -> Detection | None:
        result = self._load()(
            image_bgr,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device or None,
            verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        names = getattr(result, "names", {})
        best: Detection | None = None
        for box in boxes:
            xyxy = np.asarray(box.xyxy[0].cpu().numpy(), dtype=float).reshape(4)
            confidence = float(box.conf[0].cpu().item())
            class_id = int(box.cls[0].cpu().item())
            class_name = str(names[class_id])
            if self.target_class and class_name != self.target_class:
                continue
            candidate = Detection(
                class_name=class_name,
                confidence=confidence,
                bbox_xyxy=xyxy,
                centroid_px=np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]),
                stamp_s=stamp_s,
            )
            if best is None or candidate.confidence > best.confidence:
                best = candidate
        return best
