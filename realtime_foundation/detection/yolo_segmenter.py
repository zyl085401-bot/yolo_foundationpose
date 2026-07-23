from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo_segment_config.yml")


@dataclass
class YoloMaskResult:
  mask: np.ndarray
  class_id: int
  class_name: str
  confidence: float
  box_xyxy: np.ndarray
  area: int


class YoloSegmenter:
  def __init__(
      self,
      weights: str,
      target_class: str | None = None,
      target_class_id: int | None = None,
      conf: float = 0.35,
      imgsz: int | tuple[int, int] = 640,
      device: str | None = None,
      half: bool = True,
      min_mask_area: int = 100,
      morph_kernel: int = 5,
      input_is_rgb: bool = True,
  ):
    try:
      from ultralytics import YOLO
    except ImportError as exc:
      raise ImportError(
          "ultralytics is required for YOLO segmentation. Install it in your environment with: pip install ultralytics"
      ) from exc

    self.model = YOLO(weights, task="segment")
    self.target_class = target_class
    self.target_class_id = target_class_id
    self.conf = conf
    self.imgsz = normalize_imgsz(imgsz)
    self.device = device
    self.half = half
    self.min_mask_area = min_mask_area
    self.morph_kernel = morph_kernel
    self.input_is_rgb = input_is_rgb
    self.names = self._normalise_names(getattr(self.model, "names", {}))
    self.last_timing = {}

    if self.target_class_id is None and self.target_class is not None:
      self.target_class_id = self._class_name_to_id(self.target_class)

    self.morphology_kernel = (
      np.ones((self.morph_kernel, self.morph_kernel), dtype=np.uint8)
      if self.morph_kernel > 1
      else None
    )

  def predict_mask(self, image: np.ndarray) -> YoloMaskResult | None:
    timing = {
        "model_predict": 0.0,
        "model_preprocess": None,
        "model_inference": None,
        "model_postprocess": None,
        "model_framework_overhead": None,
        "tensor_to_cpu": 0.0,
        "mask_resize_clean": 0.0,
        "select_best_mask": 0.0,
        "total": 0.0,
    }
    total_start = time.perf_counter()
    if image is None or image.size == 0:
      self.last_timing = timing
      return None

    height, width = image.shape[:2]
    source = image[..., ::-1] if self.input_is_rgb else image
    model_start = time.perf_counter()
    prediction = self.model.predict(
        source=source,
        imgsz=self.imgsz,
        conf=self.conf,
        device=self.device,
        half=self.half,
        verbose=False,
    )[0]
    timing["model_predict"] = time.perf_counter() - model_start
    prediction_speed = getattr(prediction, "speed", {}) or {}
    model_stage_times = []
    for stage in ("preprocess", "inference", "postprocess"):
      stage_ms = prediction_speed.get(stage)
      if stage_ms is not None:
        stage_seconds = float(stage_ms) / 1000.0
        timing[f"model_{stage}"] = stage_seconds
        model_stage_times.append(stage_seconds)
    if len(model_stage_times) == 3:
      timing["model_framework_overhead"] = max(
          0.0,
          timing["model_predict"] - sum(model_stage_times),
      )

    if prediction.masks is None or prediction.boxes is None:
      timing["total"] = time.perf_counter() - total_start
      self.last_timing = timing
      return None

    tensor_start = time.perf_counter()
    masks = prediction.masks.data.detach().cpu().numpy()
    class_ids = prediction.boxes.cls.detach().cpu().numpy().astype(int)
    confidences = prediction.boxes.conf.detach().cpu().numpy()
    boxes = prediction.boxes.xyxy.detach().cpu().numpy()
    timing["tensor_to_cpu"] = time.perf_counter() - tensor_start

    best_result = None
    best_score = -1.0
    for index, raw_mask in enumerate(masks):
      select_start = time.perf_counter()
      class_id = int(class_ids[index])
      if self.target_class_id is not None and class_id != self.target_class_id:
        timing["select_best_mask"] += time.perf_counter() - select_start
        continue
      timing["select_best_mask"] += time.perf_counter() - select_start

      resize_start = time.perf_counter()
      mask = self._resize_and_clean_mask(raw_mask, width, height)
      timing["mask_resize_clean"] += time.perf_counter() - resize_start

      select_start = time.perf_counter()
      area = int(mask.sum())
      if area < self.min_mask_area:
        timing["select_best_mask"] += time.perf_counter() - select_start
        continue

      confidence = float(confidences[index])
      score = confidence * np.sqrt(area)
      if score <= best_score:
        timing["select_best_mask"] += time.perf_counter() - select_start
        continue

      best_score = score
      best_result = YoloMaskResult(
          mask=mask,
          class_id=class_id,
          class_name=self.names.get(class_id, str(class_id)),
          confidence=confidence,
          box_xyxy=boxes[index].astype(np.float32),
          area=area,
      )
      timing["select_best_mask"] += time.perf_counter() - select_start

    timing["total"] = time.perf_counter() - total_start
    self.last_timing = timing
    return best_result

  def _resize_and_clean_mask(self, raw_mask: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    mask = np.asarray(raw_mask, dtype=np.float32)
    mask_height, mask_width = mask.shape[-2:]

    gain = min(mask_width / float(width), mask_height / float(height))
    pad_width = max(mask_width - width * gain, 0.0) / 2.0
    pad_height = max(mask_height - height * gain, 0.0) / 2.0
    left = max(0, int(round(pad_width - 0.1)))
    right = min(mask_width, int(round(mask_width - pad_width + 0.1)))
    top = max(0, int(round(pad_height - 0.1)))
    bottom = min(mask_height, int(round(mask_height - pad_height + 0.1)))

    if right > left and bottom > top:
      mask = mask[top:bottom, left:right]

    if mask.shape != (height, width):
      mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (mask > 0.5).astype(np.uint8)

    if self.morphology_kernel is not None:
      mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphology_kernel)
      mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morphology_kernel)

    return mask

  def _class_name_to_id(self, class_name: str) -> int:
    for class_id, name in self.names.items():
      if name == class_name:
        return class_id
    raise ValueError(f"target_class '{class_name}' was not found in YOLO class names: {self.names}")

  @staticmethod
  def _normalise_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
      return {int(class_id): str(name) for class_id, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


def normalize_imgsz(value: Any) -> int | tuple[int, int]:
  if isinstance(value, (list, tuple)):
    if len(value) != 2:
      raise ValueError("imgsz must be an integer or [height, width]")
    height, width = (int(item) for item in value)
    if height <= 0 or width <= 0:
      raise ValueError("imgsz dimensions must be positive")
    return height, width

  size = int(value)
  if size <= 0:
    raise ValueError("imgsz must be positive")
  return size


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Test YOLO segmentation mask inference on a single image.")
  parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_FILE, help="Path to YOLO segmentation YAML config.")
  parser.add_argument("--image", type=str, default=None, help="Override test image path from YAML.")
  parser.add_argument("--weights", type=str, default=None, help="Override YOLO segmentation weights path from YAML.")
  parser.add_argument("--target-class", type=str, default=None, help="Override target class name from YAML.")
  parser.add_argument("--target-class-id", type=int, default=None, help="Override target class id from YAML.")
  parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold from YAML.")
  parser.add_argument(
      "--imgsz",
      type=int,
      nargs="+",
      default=None,
      help="Override image size with one value or height width, for example --imgsz 480 640.",
  )
  parser.add_argument("--device", type=str, default=None, help="Override device from YAML, for example cuda:0 or cpu.")
  parser.add_argument("--half", action="store_true", default=None, help="Override YAML and use FP16 inference.")
  parser.add_argument("--no-half", action="store_false", dest="half", help="Override YAML and disable FP16 inference.")
  parser.add_argument("--min-mask-area", type=int, default=None, help="Override minimum mask area from YAML.")
  parser.add_argument("--morph-kernel", type=int, default=None, help="Override morphology kernel size from YAML.")
  parser.add_argument("--output-dir", type=str, default=None, help="Override output directory from YAML.")
  parser.add_argument("--show", action="store_true", help="Show overlay preview window.")
  return parser.parse_args()


def load_yolo_config(path: str) -> dict:
  try:
    import yaml
  except ImportError as exc:
    raise ImportError("PyYAML is required to read YOLO config. Install it with: pip install pyyaml") from exc

  if not os.path.exists(path):
    raise RuntimeError(f"YOLO config file does not exist: {path}")
  with open(path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file) or {}
  if not isinstance(data, dict):
    raise RuntimeError(f"YOLO config must be a YAML mapping: {path}")
  return data


def resolve_config_path(path: str | None, config_dir: str) -> str | None:
  if path is None or os.path.isabs(path):
    return path
  return os.path.abspath(os.path.join(config_dir, path))


def build_runtime_config(args: argparse.Namespace) -> dict:
  config_path = os.path.abspath(args.config)
  config_dir = os.path.dirname(config_path)
  config = load_yolo_config(config_path)
  runtime_config = {
      "image": config.get("image"),
      "weights": config.get("weights", "../yolo_weights/best.pt"),
      "target_class": config.get("target_class"),
      "target_class_id": config.get("target_class_id"),
      "conf": float(config.get("conf", 0.35)),
      "imgsz": normalize_imgsz(config.get("imgsz", 640)),
      "device": config.get("device"),
      "half": bool(config.get("half", True)),
      "min_mask_area": int(config.get("min_mask_area", 100)),
      "morph_kernel": int(config.get("morph_kernel", 5)),
      "output_dir": config.get("output_dir", "../outputs/yolo_segment_test"),
  }

  for key in (
      "image",
      "weights",
      "target_class",
      "target_class_id",
      "conf",
      "imgsz",
      "device",
      "half",
      "min_mask_area",
      "morph_kernel",
      "output_dir",
  ):
    value = getattr(args, key)
    if value is not None:
      runtime_config[key] = value

  runtime_config["imgsz"] = normalize_imgsz(runtime_config["imgsz"])

  runtime_config["image"] = resolve_config_path(runtime_config["image"], config_dir)
  runtime_config["weights"] = resolve_config_path(runtime_config["weights"], config_dir)
  runtime_config["output_dir"] = resolve_config_path(runtime_config["output_dir"], config_dir)
  runtime_config["config_path"] = config_path
  return runtime_config


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
  import cv2

  overlay = image_bgr.copy()
  mask_bool = mask.astype(bool)
  overlay[mask_bool] = (0, 0, 255)
  return cv2.addWeighted(image_bgr, 0.65, overlay, 0.35, 0)


def run_cli(args: argparse.Namespace) -> None:
  import cv2

  config = build_runtime_config(args)
  print(f"[YOLO] Loaded config: {config['config_path']}")
  print("[YOLO] Requested inference config:")
  print(f"  image: {config['image']}")
  print(f"  weights: {config['weights']}")
  print(f"  target_class: {config['target_class']}")
  print(f"  target_class_id: {config['target_class_id']}")
  print(f"  conf: {config['conf']}")
  print(f"  imgsz: {config['imgsz']}")
  print(f"  device: {config['device']}")
  print(f"  half: {config['half']}")

  if config["image"] is None:
    raise RuntimeError("No test image configured. Set image in yolo_segment_config.yml or pass --image.")
  if not os.path.exists(config["image"]):
    raise RuntimeError(f"Test image does not exist: {config['image']}")
  if not os.path.exists(config["weights"]):
    raise RuntimeError(f"YOLO weights do not exist: {config['weights']}")

  image_bgr = cv2.imread(config["image"])
  if image_bgr is None:
    raise RuntimeError(f"Failed to read image: {config['image']}")

  detector = YoloSegmenter(
      weights=config["weights"],
      target_class=config["target_class"],
      target_class_id=config["target_class_id"],
      conf=config["conf"],
      imgsz=config["imgsz"],
      device=config["device"],
      half=config["half"],
      min_mask_area=config["min_mask_area"],
      morph_kernel=config["morph_kernel"],
      input_is_rgb=True,
  )

  image_rgb = image_bgr[..., ::-1]
  result = detector.predict_mask(image_rgb)
  if result is None:
    print("[YOLO] No target mask detected")
    return

  print("[YOLO] Detection result:")
  print(f"  class_id: {result.class_id}")
  print(f"  class_name: {result.class_name}")
  print(f"  confidence: {result.confidence:.4f}")
  print(f"  area: {result.area}")
  print(f"  box_xyxy: {result.box_xyxy.tolist()}")

  os.makedirs(config["output_dir"], exist_ok=True)
  mask_path = os.path.join(config["output_dir"], "mask.png")
  overlay_path = os.path.join(config["output_dir"], "overlay.png")
  cv2.imwrite(mask_path, (result.mask * 255).astype(np.uint8))
  overlay = make_overlay(image_bgr, result.mask)
  cv2.imwrite(overlay_path, overlay)
  print(f"[YOLO] Saved mask: {mask_path}")
  print(f"[YOLO] Saved overlay: {overlay_path}")

  if args.show:
    cv2.imshow("YOLO segmentation overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
  args = parse_args()
  try:
    run_cli(args)
  except (ImportError, RuntimeError, ValueError) as exc:
    print(f"[YOLO] {exc}", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
  main()