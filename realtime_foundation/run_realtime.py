from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import queue
import shutil
import signal
import threading
import time

import cv2
import numpy as np
import yaml

from camera.realsense_reader import RealSenseReader
from detection.yolo_segmenter import YoloSegmenter
from tracking.foundationpose_tracker import FoundationPoseRealtimeTracker


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TerminationRequested(Exception):
  pass


@dataclass(frozen=True)
class FrameMessage:
  frame_id: int
  timestamp: float
  color: np.ndarray
  depth: np.ndarray
  K: np.ndarray


@dataclass(frozen=True)
class MaskMessage:
  frame_id: int
  timestamp: float
  mask: np.ndarray
  confidence: float


@dataclass(frozen=True)
class DetectionMessage:
  frame_id: int
  timestamp: float
  color: np.ndarray
  depth: np.ndarray
  K: np.ndarray
  detection: object
  yolo_start_unix_ns: int
  yolo_end_unix_ns: int
  yolo_start_monotonic_ns: int
  yolo_end_monotonic_ns: int
  yolo_time: float
  yolo_timing: dict
  yolo_candidate_count: int


@dataclass(frozen=True)
class PoseRecord:
  filename_stem: str
  visualization: np.ndarray
  rgb: np.ndarray
  depth: np.ndarray
  K: np.ndarray
  mode: str
  frame_id: int
  timestamp: float
  yolo_mask: np.ndarray
  yolo_confidence: float
  output_pose: np.ndarray


@dataclass(frozen=True)
class TopKPoseRecord:
  filename_stem: str
  visualizations: list[np.ndarray]
  output_poses: list[np.ndarray]
  contact_sheet: np.ndarray
  summary: dict


@dataclass(frozen=True)
class RegisterQualityResult:
  accepted: bool
  reject_reason: str | None
  translation_drift: float | None
  render_iou: float | None
  top1_top2_score_gap: float | None
  rendered_mask: np.ndarray | None = None


class PoseRecordWriter:
  def __init__(self, output_dir: str, jpeg_quality: int, config: dict, reset_on_start: bool = True):
    self.output_dir = output_dir
    self.rgb_dir = os.path.join(output_dir, "matched_rgb")
    self.frame_data_dir = os.path.join(output_dir, "frame_data")
    self.topk_dir = os.path.join(output_dir, "top5_poses")
    self.run_id = time.strftime("run_%Y%m%d_%H%M%S")
    self.jpeg_quality = max(1, min(int(jpeg_quality), 100))
    self.tasks = queue.Queue()
    self.thread = threading.Thread(target=self._write_loop, name="pose-record-writer", daemon=True)
    self.write_errors = 0

    os.makedirs(output_dir, exist_ok=True)
    if reset_on_start:
      for path in (self.rgb_dir, self.frame_data_dir, self.topk_dir):
        if os.path.isdir(path):
          shutil.rmtree(path)
    os.makedirs(self.rgb_dir, exist_ok=True)
    os.makedirs(self.frame_data_dir, exist_ok=True)
    os.makedirs(self.topk_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as file:
      yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)
    with open(os.path.join(self.topk_dir, f"{self.run_id}_config.yaml"), "w", encoding="utf-8") as file:
      yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)

  def start(self) -> None:
    self.thread.start()

  def submit(self, record: PoseRecord | TopKPoseRecord) -> None:
    self.tasks.put(record)

  def stop(self) -> None:
    self.tasks.put(None)
    self.thread.join()
    if self.write_errors:
      print(f"[Recorder] Finished with {self.write_errors} write error(s)")

  def _write_loop(self) -> None:
    while True:
      record = self.tasks.get()
      try:
        if record is None:
          return
        if isinstance(record, TopKPoseRecord):
          self._write_topk_record(record)
        else:
          self._write_record(record)
      except Exception as exc:
        self.write_errors += 1
        filename = getattr(record, "filename_stem", "unknown")
        print(f"[Recorder] Failed to save {filename}: {exc}")
      finally:
        self.tasks.task_done()

  def _write_record(self, record: PoseRecord) -> None:
    image_path = os.path.join(self.rgb_dir, f"{record.filename_stem}.jpg")
    replay_path = os.path.join(self.frame_data_dir, f"{record.filename_stem}.npz")
    image_ok = cv2.imwrite(
        image_path,
        record.visualization[..., ::-1],
        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
    )
    if not image_ok:
      raise RuntimeError(f"cv2.imwrite failed: {image_path}")
    np.savez(
        replay_path,
        format_version=np.asarray(1, dtype=np.int32),
        mode=np.asarray(record.mode),
        frame_id=np.asarray(record.frame_id, dtype=np.int64),
        timestamp=np.asarray(record.timestamp, dtype=np.float64),
        rgb=record.rgb,
        depth=record.depth,
        K=record.K,
        yolo_mask=record.yolo_mask,
        yolo_confidence=np.asarray(record.yolo_confidence, dtype=np.float32),
        output_pose=record.output_pose,
    )

  def _write_topk_record(self, record: TopKPoseRecord) -> None:
    filename_stem = f"{self.run_id}_{record.filename_stem}"
    for rank, (visualization, pose) in enumerate(zip(record.visualizations, record.output_poses), start=1):
      rank_stem = f"{filename_stem}_rank_{rank:02d}"
      image_path = os.path.join(self.topk_dir, f"{rank_stem}.jpg")
      if not cv2.imwrite(
          image_path,
          visualization[..., ::-1],
          [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
      ):
        raise RuntimeError(f"cv2.imwrite failed: {image_path}")
      np.savetxt(os.path.join(self.topk_dir, f"{rank_stem}_pose.txt"), pose)

    contact_path = os.path.join(self.topk_dir, f"{filename_stem}_contact_sheet.jpg")
    if not cv2.imwrite(
        contact_path,
        record.contact_sheet[..., ::-1],
        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
    ):
      raise RuntimeError(f"cv2.imwrite failed: {contact_path}")
    with open(os.path.join(self.topk_dir, f"{filename_stem}_summary.json"), "w", encoding="utf-8") as file:
      json.dump(record.summary, file, indent=2, ensure_ascii=False)


class LatestTopic:
  def __init__(self, name: str):
    self.name = name
    self.condition = threading.Condition()
    self.message = None
    self.sequence = 0

  def publish(self, message) -> int:
    with self.condition:
      self.sequence += 1
      self.message = message
      self.condition.notify_all()
      return self.sequence

  def get_latest(self):
    with self.condition:
      return self.message

  def get_snapshot(self):
    with self.condition:
      return self.sequence, self.message

  def wait_for_newer(
      self,
      after_sequence: int,
      stop_event: threading.Event | None = None,
      timeout: float | None = None,
  ):
    with self.condition:
      self.condition.wait_for(
          lambda: self.sequence > after_sequence or (stop_event is not None and stop_event.is_set()),
          timeout=timeout,
      )
      if self.sequence <= after_sequence:
        return None
      return self.sequence, self.message

  def wake_waiters(self) -> None:
    with self.condition:
      self.condition.notify_all()


class RgbdImagePublisher:
  def __init__(self, camera):
    self.camera = camera
    self.image_topic = LatestTopic("/camera/rgbd/latest")
    self.stop_event = threading.Event()
    self.thread = threading.Thread(target=self._capture_loop, name="rgbd-image-publisher", daemon=True)
    self.latest_frame_id = 0
    self.error_lock = threading.Lock()
    self.error = None

  def start(self) -> None:
    with self.error_lock:
      self.error = None
    self.camera.start()
    self.thread.start()

  def stop(self) -> None:
    self.stop_event.set()
    self.camera.stop()
    if self.thread.is_alive():
      self.thread.join(timeout=2.0)

  def get_latest(self) -> FrameMessage | None:
    with self.error_lock:
      error = self.error
    if error is not None:
      raise RuntimeError(f"RGB-D capture thread failed: {error}") from error
    return self.image_topic.get_latest()

  def _capture_loop(self) -> None:
    try:
      while not self.stop_event.is_set():
        frame = self.camera.get_frame()
        if frame is None:
          continue
        if len(frame) == 3:
          color, depth, K = frame
          timestamp = time.time()
        elif len(frame) == 4:
          color, depth, K, timestamp = frame
        else:
          raise RuntimeError(f"Camera reader returned {len(frame)} values; expected 3 or 4")
        self.latest_frame_id += 1
        self.image_topic.publish(FrameMessage(
            frame_id=self.latest_frame_id,
            timestamp=float(timestamp),
            color=color,
            depth=depth,
            K=K,
        ))
    except Exception as exc:
      if not self.stop_event.is_set():
        with self.error_lock:
          self.error = exc
        self.stop_event.set()


class YoloDetectionWorker:
  def __init__(
      self,
      detector: YoloSegmenter,
      image_topic: LatestTopic,
      frame_stride: int = 1,
  ):
    self.detector = detector
    self.image_topic = image_topic
    self.detection_topic = LatestTopic("/yolo/detection/latest")
    self.frame_stride = max(1, int(frame_stride))
    self.stop_event = threading.Event()
    self.pause_event = threading.Event()
    self.inference_lock = threading.Lock()
    self.candidate_condition = threading.Condition()
    self.candidate_search_enabled = True
    self.pending_candidate_sequence = None
    self.resolved_candidate_sequence = 0
    self.error_lock = threading.Lock()
    self.error = None
    self.thread = threading.Thread(target=self._detect_loop, name="yolo-detection-worker", daemon=True)
    self.last_processed_frame_id = 0

  def start(self) -> None:
    self.thread.start()

  def stop(self) -> None:
    self.stop_event.set()
    self.image_topic.wake_waiters()
    self.detection_topic.wake_waiters()
    with self.candidate_condition:
      self.candidate_condition.notify_all()
    if self.thread.is_alive():
      self.thread.join(timeout=2.0)
    if self.thread.is_alive():
      print("[Realtime] Warning: YOLO detection thread did not stop within 2.0 s")

  def pause(self) -> None:
    self.pause_event.set()
    with self.inference_lock:
      pass

  def resume(self) -> None:
    self.pause_event.clear()

  def get_latest(self) -> DetectionMessage | None:
    self.raise_if_failed()
    return self.detection_topic.get_latest()

  def get_latest_snapshot(self) -> tuple[int, DetectionMessage | None]:
    self.raise_if_failed()
    return self.detection_topic.get_snapshot()

  def wait_for_detection(
      self,
      after_sequence: int,
      timeout: float | None = None,
  ) -> tuple[int, DetectionMessage] | None:
    snapshot = self.detection_topic.wait_for_newer(
        after_sequence,
        stop_event=self.stop_event,
        timeout=timeout,
    )
    self.raise_if_failed()
    return snapshot

  def resume_candidate_search(self, candidate_sequence: int) -> None:
    with self.candidate_condition:
      self.resolved_candidate_sequence = max(self.resolved_candidate_sequence, candidate_sequence)
      if self.pending_candidate_sequence == candidate_sequence:
        self.pending_candidate_sequence = None
      self.candidate_condition.notify_all()

  def finish_candidate_search(self, candidate_sequence: int) -> None:
    with self.candidate_condition:
      self.candidate_search_enabled = False
      self.resolved_candidate_sequence = max(self.resolved_candidate_sequence, candidate_sequence)
      self.pending_candidate_sequence = None
      self.candidate_condition.notify_all()

  def enable_candidate_search(self) -> int:
    with self.candidate_condition:
      self.candidate_search_enabled = True
    sequence, _ = self.detection_topic.get_snapshot()
    return sequence

  def raise_if_failed(self) -> None:
    with self.error_lock:
      error = self.error
    if error is not None:
      raise RuntimeError(f"YOLO detection thread failed: {error}") from error

  def _detect_loop(self) -> None:
    try:
      while not self.stop_event.is_set():
        if self.pause_event.is_set():
          time.sleep(0.001)
          continue
        frame_message = self.image_topic.get_latest()
        if frame_message is None or frame_message.frame_id == self.last_processed_frame_id:
          time.sleep(0.001)
          continue
        self.last_processed_frame_id = frame_message.frame_id
        if frame_message.frame_id % self.frame_stride != 0:
          continue

        with self.inference_lock:
          if self.pause_event.is_set():
            continue
          yolo_start_unix_ns = time.time_ns()
          yolo_start_monotonic_ns = time.perf_counter_ns()
          detection = self.detector.predict_mask(frame_message.color)
          yolo_end_monotonic_ns = time.perf_counter_ns()
          yolo_end_unix_ns = time.time_ns()
          yolo_time = (yolo_end_monotonic_ns - yolo_start_monotonic_ns) / 1_000_000_000.0
          publication_sequence = self.detection_topic.publish(DetectionMessage(
              frame_id=frame_message.frame_id,
              timestamp=frame_message.timestamp,
              color=frame_message.color,
              depth=frame_message.depth,
              K=frame_message.K,
              detection=detection,
              yolo_start_unix_ns=yolo_start_unix_ns,
              yolo_end_unix_ns=yolo_end_unix_ns,
              yolo_start_monotonic_ns=yolo_start_monotonic_ns,
              yolo_end_monotonic_ns=yolo_end_monotonic_ns,
              yolo_time=yolo_time,
              yolo_timing=dict(getattr(self.detector, "last_timing", {})),
              yolo_candidate_count=int(getattr(self.detector, "last_candidate_count", 0)),
          ))

        if detection is not None:
          self._wait_for_candidate_resolution(publication_sequence)
    except Exception as exc:
      if not self.stop_event.is_set():
        with self.error_lock:
          self.error = exc
        self.stop_event.set()
        self.detection_topic.wake_waiters()
        with self.candidate_condition:
          self.candidate_condition.notify_all()

  def _wait_for_candidate_resolution(self, publication_sequence: int) -> None:
    with self.candidate_condition:
      if not self.candidate_search_enabled:
        return
      self.pending_candidate_sequence = publication_sequence
      self.candidate_condition.wait_for(
          lambda: (
              self.stop_event.is_set()
              or not self.candidate_search_enabled
              or self.resolved_candidate_sequence >= publication_sequence
          )
      )
      if self.pending_candidate_sequence == publication_sequence:
        self.pending_candidate_sequence = None


def load_config(path: str) -> dict:
  with open(path, "r", encoding="utf-8") as file:
    return yaml.safe_load(file)


def resolve_path(path: str | None) -> str | None:
  if path is None or os.path.isabs(path):
    return path
  return os.path.abspath(os.path.join(REPO_ROOT, path))


def build_camera_reader(
  camera_cfg: dict,
  config_path: str | Path,
  source_override: str | None = None,
):
  source = str(source_override or camera_cfg.get("source", "realsense")).strip().lower()
  if source == "realsense":
    return RealSenseReader(
        width=int(camera_cfg.get("width", 640)),
        height=int(camera_cfg.get("height", 480)),
        fps=int(camera_cfg.get("fps", 30)),
        serial=camera_cfg.get("serial"),
        depth_min=float(camera_cfg.get("depth_min", 0.001)),
        depth_max=float(camera_cfg.get("depth_max", 3.0)),
        align_to_color=bool(camera_cfg.get("align_to_color", True)),
        reset_before_start=bool(camera_cfg.get("reset_before_start", True)),
    )
  if source == "ros2":
    try:
      from camera.ros2_rgbd_shm_reader import Ros2RgbdSharedMemoryReader
    except ImportError as exc:
      raise RuntimeError(
          "ROS 2 camera input requires the ROS-derived image and a sourced ROS environment. "
          "Run with FoundationPose/docker/run_container_jetson_ros2.sh."
      ) from exc
    return Ros2RgbdSharedMemoryReader.from_config(
        config_path=Path(config_path),
        camera_config=camera_cfg,
        ros2_config=camera_cfg.get("ros2", {}) or {},
    )
  raise ValueError(f"Unsupported camera.source {source!r}; expected 'realsense' or 'ros2'")


def resolve_tensorrt_backends(tracker_cfg: dict) -> dict:
  backends = dict(tracker_cfg.get("tensorrt", {}))
  refiner = dict(backends.get("refiner", {}))
  scorer = dict(backends.get("scorer", {}))
  refiner["engine_path"] = resolve_path(refiner.get("engine_path"))
  refiner["engine_paths"] = {
      str(network_stage): resolve_path(engine_path)
      for network_stage, engine_path in dict(refiner.get("engine_paths", {})).items()
  }
  scorer["engine_paths"] = {
      int(candidate_count): resolve_path(engine_path)
      for candidate_count, engine_path in dict(scorer.get("engine_paths", {})).items()
  }
  return {"refiner": refiner, "scorer": scorer}


def resolve_render_lod_config(tracker_cfg: dict) -> dict:
  render_lod = dict(tracker_cfg.get("render_lod", {}) or {})
  render_lod["mesh_file"] = resolve_path(render_lod.get("mesh_file"))
  comparison = dict(render_lod.get("comparison", {}) or {})
  comparison["output_dir"] = resolve_path(
      comparison.get("output_dir", "realtime_foundation/outputs/render_lod_comparison")
  )
  render_lod["comparison"] = comparison
  return render_lod


def build_tracker(tracker_cfg: dict, candidate_pipeline_debug_enabled: bool = False) -> FoundationPoseRealtimeTracker:
  return FoundationPoseRealtimeTracker(
      mesh_file=resolve_path(tracker_cfg["mesh_file"]),
      debug_dir=resolve_path(tracker_cfg.get("debug_dir", "realtime_foundation/outputs/frame_records")),
      debug=int(tracker_cfg.get("debug", 1)),
      use_float32_crop_window_patch=bool(tracker_cfg.get("use_float32_crop_window_patch", True)),
      init_min_n_views=int(tracker_cfg.get("init_min_n_views", 40)),
      init_inplane_step=int(tracker_cfg.get("init_inplane_step", 60)),
      est_refine_iter=int(tracker_cfg.get("est_refine_iter", 5)),
      init_strategy=tracker_cfg.get("init_strategy", "default"),
      coarse_refine_iter=int(tracker_cfg.get("coarse_refine_iter", 1)),
      coarse_score_filter=tracker_cfg.get("coarse_score_filter", "none"),
      coarse_score_top_k=int(tracker_cfg.get("coarse_score_top_k", 999999)),
      fine_refine_iter=int(tracker_cfg.get("fine_refine_iter", 2)),
      fine_top_k=int(tracker_cfg.get("fine_top_k", 16)),
      skip_redundant_coarse_scorer=bool(tracker_cfg.get("skip_redundant_coarse_scorer", False)),
      axis_prior_filter=tracker_cfg.get("axis_prior_filter", "none") if bool(tracker_cfg.get("axis_prior_enabled", False)) else "none",
      axis_prior_model_axis=tuple(float(value) for value in tracker_cfg.get("axis_prior_model_axis", [0.0, 0.0, 1.0])),
      axis_prior_max_angle_deg=float(tracker_cfg.get("axis_prior_max_angle_deg", 45.0)),
      axis_prior_min_candidates=int(tracker_cfg.get("axis_prior_min_candidates", 12)),
      axis_prior_max_candidates=int(tracker_cfg.get("axis_prior_max_candidates", 0)),
      axis_prior_min_points=int(tracker_cfg.get("axis_prior_min_points", 500)),
      axis_prior_min_confidence=float(tracker_cfg.get("axis_prior_min_confidence", 1.4)),
      axis_prior_visualization_enabled=bool(tracker_cfg.get("axis_prior_visualization_enabled", False)),
      axis_prior_visualization_dir=resolve_path(tracker_cfg.get("axis_prior_visualization_dir", "realtime_foundation/outputs/axis_prior_debug")),
      axis_prior_visualization_top_n=int(tracker_cfg.get("axis_prior_visualization_top_n", 24)),
      axis_prior_visualization_boundary_margin=int(tracker_cfg.get("axis_prior_visualization_boundary_margin", 6)),
      axis_prior_visualization_max_records=int(tracker_cfg.get("axis_prior_visualization_max_records", 20)),
      network_input_capture={
          **dict(tracker_cfg.get("network_input_capture", {})),
          "output_dir": resolve_path(dict(tracker_cfg.get("network_input_capture", {})).get("output_dir", "realtime_foundation/outputs/network_input_capture")),
      },
      render_profile_enabled=bool(tracker_cfg.get("render_profile_enabled", False)),
      render_batched_matmul_enabled=bool(tracker_cfg.get("render_batched_matmul_enabled", False)),
      scorer_precomputed_xyz_enabled=bool(tracker_cfg.get("scorer_precomputed_xyz_enabled", False)),
      refiner_stage1_optimizations_enabled=bool(tracker_cfg.get("refiner_stage1_optimizations_enabled", False)),
      refiner_shared_warp_grid_enabled=bool(tracker_cfg.get("refiner_shared_warp_grid_enabled", False)),
      scorer_shared_warp_grid_enabled=bool(tracker_cfg.get("scorer_shared_warp_grid_enabled", False)),
      scorer_skip_unused_depth_warp_enabled=bool(tracker_cfg.get("scorer_skip_unused_depth_warp_enabled", False)),
      tensorrt_backends=resolve_tensorrt_backends(tracker_cfg),
      refiner_input_sizes=dict(tracker_cfg.get("refiner_input_sizes", {})),
      track_refine_iter=int(tracker_cfg.get("track_refine_iter", 2)),
      vis_mode=tracker_cfg.get("vis_mode", "box"),
      contour_thickness=int(tracker_cfg.get("contour_thickness", 3)),
      axis_scale=float(tracker_cfg.get("axis_scale", 0.1)),
      network_internal_sync_enabled=bool(tracker_cfg.get("network_internal_sync_enabled", True)),
      frame_statistics_reuse_enabled=bool(tracker_cfg.get("frame_statistics_reuse_enabled", False)),
      quality_render_mask_reuse_enabled=bool(tracker_cfg.get("quality_render_mask_reuse_enabled", False)),
      render_lod=resolve_render_lod_config(tracker_cfg),
      candidate_pipeline_debug_enabled=bool(candidate_pipeline_debug_enabled),
  )


def add_pose_record_label(vis: np.ndarray, frame_id: int, mode: str, confidence: float | None) -> np.ndarray:
  labeled = vis.copy()
  confidence_text = "N/A" if confidence is None else f"{confidence:.3f}"
  text = f"frame={frame_id} mode={mode} yolo_conf={confidence_text}"
  cv2.putText(labeled, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
  cv2.putText(labeled, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
  return labeled


def array_to_numpy(value) -> np.ndarray:
  if hasattr(value, "detach"):
    value = value.detach().cpu().numpy()
  return np.asarray(value)


def candidate_axis_angle_deg(centered_pose: np.ndarray, model_axis, scene_axis) -> float | None:
  if scene_axis is None:
    return None
  model_axis = np.asarray(model_axis, dtype=np.float32).reshape(3)
  scene_axis = np.asarray(scene_axis, dtype=np.float32).reshape(3)
  model_norm = float(np.linalg.norm(model_axis))
  scene_norm = float(np.linalg.norm(scene_axis))
  if model_norm < 1e-8 or scene_norm < 1e-8:
    return None
  candidate_axis = centered_pose[:3, :3] @ (model_axis / model_norm)
  alignment = float(np.clip(abs(np.dot(candidate_axis, scene_axis / scene_norm)), 0.0, 1.0))
  return float(np.degrees(np.arccos(alignment)))


def add_topk_label(vis: np.ndarray, rank: int, score: float, angle: float | None, rendered_iou: float) -> np.ndarray:
  labeled = vis.copy()
  lines = [
      f"rank={rank} score={score:.6f}",
      f"axis_angle={'N/A' if angle is None else f'{angle:.2f} deg'} render_iou={rendered_iou:.3f}",
  ]
  for line_index, text in enumerate(lines):
    y = 28 + line_index * 26
    cv2.putText(labeled, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(labeled, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
  return labeled


def make_contact_sheet(images: list[np.ndarray], columns: int = 3, thumbnail_width: int = 480) -> np.ndarray:
  if not images:
    return np.zeros((1, 1, 3), dtype=np.uint8)
  source_height, source_width = images[0].shape[:2]
  thumbnail_height = max(1, int(round(source_height * thumbnail_width / source_width)))
  rows = (len(images) + columns - 1) // columns
  sheet = np.zeros((rows * thumbnail_height, columns * thumbnail_width, 3), dtype=np.uint8)
  for index, image in enumerate(images):
    thumbnail = cv2.resize(image, (thumbnail_width, thumbnail_height), interpolation=cv2.INTER_AREA)
    row, column = divmod(index, columns)
    sheet[
        row * thumbnail_height:(row + 1) * thumbnail_height,
        column * thumbnail_width:(column + 1) * thumbnail_width,
    ] = thumbnail
  return sheet


def build_topk_pose_record(
    tracker: FoundationPoseRealtimeTracker,
    color: np.ndarray,
    K: np.ndarray,
  target_mask: np.ndarray,
    frame_id: int,
    count: int,
) -> TopKPoseRecord | None:
  estimator = tracker.estimator
  ranked_poses = getattr(estimator, "poses", None)
  ranked_scores = getattr(estimator, "scores", None)
  if ranked_poses is None or ranked_scores is None:
    return None

  centered_poses = array_to_numpy(ranked_poses).astype(np.float32)
  scores = array_to_numpy(ranked_scores).reshape(-1).astype(np.float32)
  count = min(max(1, int(count)), len(centered_poses), len(scores))
  if count <= 0:
    return None

  timing = dict(getattr(estimator, "last_register_timing", {}))
  scene_axis = timing.get("axis_prior_scene_axis")
  axis_enabled = tracker.axis_prior_filter == "depth_pca"
  axis_tag = "axis_on" if axis_enabled else "axis_off"
  tf_to_centered_mesh = array_to_numpy(estimator.get_tf_to_centered_mesh()).astype(np.float32).reshape(4, 4)
  target_mask = np.asarray(target_mask, dtype=np.uint8) > 0
  visualizations = []
  output_poses = []
  ranks = []

  for index in range(count):
    centered_pose = centered_poses[index].reshape(4, 4)
    output_pose = centered_pose @ tf_to_centered_mesh
    rendered_mask = tracker.render_pose_mask(K, color.shape[:2], pose=centered_pose) > 0
    intersection = int(np.logical_and(rendered_mask, target_mask).sum())
    union = int(np.logical_or(rendered_mask, target_mask).sum())
    rendered_iou = float(intersection / union) if union > 0 else 0.0
    axis_angle = candidate_axis_angle_deg(
        centered_pose,
        tracker.axis_prior_model_axis,
        scene_axis,
    )
    visualization = tracker.draw_visualization(
        color,
        K,
        output_pose,
        centered_pose=centered_pose,
    )
    mask_pixels = target_mask
    visualization[mask_pixels] = (
        0.65 * visualization[mask_pixels] + 0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)
    score = float(scores[index])
    visualization = add_topk_label(visualization, index + 1, score, axis_angle, rendered_iou)
    visualizations.append(visualization)
    output_poses.append(output_pose.copy())
    ranks.append({
        "rank": index + 1,
        "score": score,
        "axis_angle_deg": axis_angle,
        "rendered_mask_iou": rendered_iou,
        "centered_pose": centered_pose.tolist(),
        "output_pose": output_pose.tolist(),
    })

  filename_stem = f"frame_{frame_id:06d}_{axis_tag}"
  summary = {
      "frame_id": int(frame_id),
      "axis_prior_enabled": axis_enabled,
      "axis_prior_filter": timing.get("axis_prior_filter"),
      "axis_prior_status": timing.get("axis_prior_status"),
      "axis_prior_points": int(timing.get("axis_prior_points", 0)),
      "axis_prior_confidence": timing.get("axis_prior_confidence"),
      "axis_prior_eigenvalues": timing.get("axis_prior_eigenvalues"),
      "axis_prior_scene_axis": scene_axis,
      "axis_prior_candidates_before": int(timing.get("axis_prior_candidates_before", 0)),
      "axis_prior_candidates_after": int(timing.get("axis_prior_candidates_after", 0)),
      "axis_prior_max_angle_deg": float(tracker.axis_prior_max_angle_deg),
      "axis_prior_min_candidates": int(tracker.axis_prior_min_candidates),
      "axis_prior_max_candidates": int(tracker.axis_prior_max_candidates),
      "ranks": ranks,
  }
  return TopKPoseRecord(
      filename_stem=filename_stem,
      visualizations=visualizations,
      output_poses=output_poses,
      contact_sheet=make_contact_sheet(visualizations),
      summary=summary,
  )


def should_check_detection(frame_index: int, interval: int) -> bool:
  return interval > 0 and frame_index % interval == 0


def save_pose(path: str, pose: np.ndarray) -> None:
  output_dir = os.path.dirname(path)
  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
  np.savetxt(path, pose.reshape(4, 4))


def estimate_mask_translation(depth: np.ndarray, mask: np.ndarray, K: np.ndarray) -> list[float] | None:
  ys, xs = np.where(mask > 0)
  if len(xs) == 0:
    return None
  valid = (mask > 0) & (depth >= 0.001)
  if not valid.any():
    return None
  uc = (xs.min() + xs.max()) / 2.0
  vc = (ys.min() + ys.max()) / 2.0
  zc = float(np.median(depth[valid]))
  center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1.0]).reshape(3, 1)) * zc
  return [float(value) for value in center.reshape(3)]


def tensor_or_array_to_list(value, limit: int | None = None) -> list[float]:
  if value is None:
    return []
  if hasattr(value, "detach"):
    array = value.detach().cpu().numpy()
  else:
    array = np.asarray(value)
  array = np.asarray(array).reshape(-1)
  if limit is not None:
    array = array[:limit]
  return [float(item) for item in array]


def top1_top2_score_gap(tracker: FoundationPoseRealtimeTracker) -> float | None:
  scores_top2 = tensor_or_array_to_list(getattr(tracker.estimator, "scores", None), limit=2)
  if len(scores_top2) < 2:
    return None
  return float(scores_top2[0] - scores_top2[1])


def estimator_pose_last_translation(tracker: FoundationPoseRealtimeTracker) -> np.ndarray | None:
  pose_last = getattr(tracker.estimator, "pose_last", None)
  if pose_last is None:
    return None
  if hasattr(pose_last, "detach"):
    pose_last = pose_last.detach().cpu().numpy()
  pose_last = np.asarray(pose_last, dtype=np.float32).reshape(4, 4)
  return pose_last[:3, 3]


def evaluate_register_quality(
    tracker: FoundationPoseRealtimeTracker,
    pose: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    runtime_cfg: dict,
) -> RegisterQualityResult:
  max_translation_drift = float(runtime_cfg.get("register_max_translation_drift", 0.0))
  min_render_iou = float(runtime_cfg.get("register_min_render_iou", 0.0))
  uncertain_score_gap = float(runtime_cfg.get("register_uncertain_score_gap", 0.0))
  uncertain_render_iou = float(runtime_cfg.get("register_uncertain_render_iou", 0.0))
  translation_drift = None
  render_iou = None
  rendered_mask = None

  mask_translation = estimate_mask_translation(depth, mask, K)
  if max_translation_drift > 0 and mask_translation is not None:
    mask_translation_array = np.asarray(mask_translation, dtype=np.float32)
    returned_pose_translation = np.asarray(pose, dtype=np.float32).reshape(4, 4)[:3, 3]
    centered_pose_translation = estimator_pose_last_translation(tracker)
    pose_translation = centered_pose_translation if centered_pose_translation is not None else returned_pose_translation
    translation_drift = float(np.linalg.norm(pose_translation - mask_translation_array))
    if translation_drift > max_translation_drift:
      return RegisterQualityResult(
          accepted=False,
          reject_reason="translation_drift_above_threshold",
          translation_drift=translation_drift,
          render_iou=None,
          top1_top2_score_gap=top1_top2_score_gap(tracker),
      )

  needs_render_iou = min_render_iou > 0 or (uncertain_score_gap > 0 and uncertain_render_iou > 0)
  if needs_render_iou:
    rendered_mask = tracker.render_pose_mask(K, depth.shape[:2])
    render_iou = tracker.mask_iou(K, depth.shape[:2], mask, rendered_mask=rendered_mask)
  score_gap = top1_top2_score_gap(tracker)

  if min_render_iou > 0 and render_iou is not None and render_iou < min_render_iou:
    return RegisterQualityResult(
        accepted=False,
        reject_reason="rendered_mask_iou_below_threshold",
        translation_drift=translation_drift,
        render_iou=render_iou,
        top1_top2_score_gap=score_gap,
        rendered_mask=rendered_mask,
    )

  if (
      uncertain_score_gap > 0
      and uncertain_render_iou > 0
      and score_gap is not None
      and score_gap <= uncertain_score_gap
      and render_iou is not None
      and render_iou < uncertain_render_iou
  ):
    return RegisterQualityResult(
        accepted=False,
        reject_reason="ambiguous_score_gap_and_render_iou",
        translation_drift=translation_drift,
        render_iou=render_iou,
        top1_top2_score_gap=score_gap,
        rendered_mask=rendered_mask,
    )

  return RegisterQualityResult(
      accepted=True,
      reject_reason=None,
      translation_drift=translation_drift,
      render_iou=render_iou,
      top1_top2_score_gap=score_gap,
      rendered_mask=rendered_mask,
  )


def count_valid_depth_pixels(depth: np.ndarray, mask: np.ndarray, min_depth: float = 0.001) -> int:
  return int(((mask > 0) & (depth >= min_depth)).sum())


def mask_center(mask: np.ndarray) -> tuple[float, float] | None:
  ys, xs = np.where(mask > 0)
  if len(xs) == 0:
    return None
  return float(xs.mean()), float(ys.mean())


def detection_reject_reason(detection, depth: np.ndarray, runtime_cfg: dict) -> str | None:
  min_conf = float(runtime_cfg.get("min_detection_conf", 0.0))
  if detection.confidence < min_conf:
    return f"confidence {detection.confidence:.3f} < {min_conf:.3f}"

  min_area = int(runtime_cfg.get("min_detection_area", 0))
  max_area = int(runtime_cfg.get("max_detection_area", 0))
  if detection.area < min_area:
    return f"mask area {detection.area} < {min_area}"
  if max_area > 0 and detection.area > max_area:
    return f"mask area {detection.area} > {max_area}"

  center = mask_center(detection.mask)
  if center is None:
    return "empty mask"
  center_x, center_y = center
  roi = runtime_cfg.get("detection_roi_xyxy")
  if roi is not None:
    x1, y1, x2, y2 = [float(value) for value in roi]
    if not (x1 <= center_x <= x2 and y1 <= center_y <= y2):
      return f"mask center ({center_x:.1f}, {center_y:.1f}) outside ROI {roi}"

  valid_depth = depth[(detection.mask > 0) & (depth >= 0.001)]
  if valid_depth.size == 0:
    return "no valid depth inside mask"
  median_depth = float(np.median(valid_depth))
  min_median_depth = float(runtime_cfg.get("min_mask_median_depth", 0.0))
  max_median_depth = float(runtime_cfg.get("max_mask_median_depth", 0.0))
  if min_median_depth > 0 and median_depth < min_median_depth:
    return f"median depth {median_depth:.3f} < {min_median_depth:.3f}"
  if max_median_depth > 0 and median_depth > max_median_depth:
    return f"median depth {median_depth:.3f} > {max_median_depth:.3f}"

  return None


def detection_border_reject_reason(detection, image_shape: tuple[int, int], margin: int) -> str | None:
  if margin <= 0:
    return None

  height, width = image_shape
  x1, y1, x2, y2 = [float(value) for value in detection.box_xyxy]
  if x1 < margin:
    return f"bbox left edge {x1:.1f} < margin {margin}"
  if y1 < margin:
    return f"bbox top edge {y1:.1f} < margin {margin}"
  if x2 > width - margin:
    return f"bbox right edge {x2:.1f} > width-margin {width - margin}"
  if y2 > height - margin:
    return f"bbox bottom edge {y2:.1f} > height-margin {height - margin}"
  return None


def register_stability_reject_reason(
    detection,
    stable_areas: list[int],
    required_count: int,
    max_area_change: float,
) -> str | None:
  required_count = max(1, required_count)
  stable_areas.append(int(detection.area))
  if len(stable_areas) > required_count:
    del stable_areas[0:len(stable_areas) - required_count]

  if len(stable_areas) < required_count:
    return f"stable detections {len(stable_areas)}/{required_count}"

  if required_count <= 1 or max_area_change <= 0:
    return None

  min_area = min(stable_areas)
  max_area = max(stable_areas)
  area_change = (max_area - min_area) / max(max_area, 1)
  if area_change > max_area_change:
    return f"mask area change {area_change:.2f} > {max_area_change:.2f} across recent detections"

  return None


def should_log_status(frame_index: int, interval: int) -> bool:
  return interval > 0 and frame_index % interval == 0


def format_seconds(value, fallback: str = "N/A") -> str:
  if value is None:
    return fallback
  return f"{float(value):.3f}s"


def seconds_to_ms(value):
  if value is None:
    return None
  return float(value) * 1000.0


def foundation_register_timing_rows(timing: dict, register_wall_time: float | None = None) -> list[tuple[str, float | str | None]]:
  refiner_coarse_detail = timing.get("refiner_coarse_detail", {})
  refiner_fine_detail = timing.get("refiner_fine_detail", {})
  scorer_coarse_detail = timing.get("scorer_coarse_detail", {})
  scorer_fine_detail = timing.get("scorer_fine_detail", {})
  scorer_coarse_time = timing.get("scorer_coarse")
  scorer_coarse_ran = scorer_coarse_time is not None and float(scorer_coarse_time) > 0.0
  scorer_fine_detail_indent = "      " if scorer_coarse_ran else "    "
  register_time = timing.get("register", register_wall_time)
  rows = [
      ("foundation_total", seconds_to_ms(register_time)),
      ("  foundation_depth_preprocess", seconds_to_ms(timing.get("depth_preprocess"))),
      ("  foundation_frame_statistics_reuse", timing.get("frame_statistics_reuse_status")),
      ("  foundation_frame_statistics", seconds_to_ms(timing.get("frame_statistics"))),
      ("  foundation_pose_hypothesis", seconds_to_ms(timing.get("pose_hypothesis"))),
      ("  foundation_axis_prior", seconds_to_ms(timing.get("axis_prior"))),
      ("  foundation_frame_to_cuda", seconds_to_ms(timing.get("frame_to_cuda"))),
      ("  foundation_refiner", seconds_to_ms(timing.get("refiner"))),
      ("    refiner_coarse", seconds_to_ms(timing.get("refiner_coarse"))),
      ("      refiner_coarse_backend", refiner_coarse_detail.get("network_backend")),
      ("      refiner_coarse_network_internal_sync", refiner_coarse_detail.get("network_internal_sync_status")),
      ("      refiner_coarse_fallback", refiner_coarse_detail.get("fallback_reason")),
      ("      refiner_coarse_crop_window", seconds_to_ms(refiner_coarse_detail.get("crop_window"))),
      ("      refiner_coarse_render", seconds_to_ms(refiner_coarse_detail.get("render"))),
      ("      refiner_coarse_render_postprocess", seconds_to_ms(refiner_coarse_detail.get("render_postprocess"))),
      ("      refiner_coarse_warp", seconds_to_ms(refiner_coarse_detail.get("warp"))),
      ("      refiner_coarse_transform", seconds_to_ms(refiner_coarse_detail.get("transform"))),
      ("      refiner_coarse_input_pack", seconds_to_ms(refiner_coarse_detail.get("input_pack"))),
      ("      refiner_coarse_network_forward", seconds_to_ms(refiner_coarse_detail.get("network_forward"))),
      ("        refiner_coarse_layout_convert", seconds_to_ms(refiner_coarse_detail.get("layout_convert"))),
      ("        refiner_coarse_dtype_convert", seconds_to_ms(refiner_coarse_detail.get("dtype_convert"))),
      ("        refiner_coarse_tensorrt_execute", seconds_to_ms(refiner_coarse_detail.get("tensorrt_execute"))),
      ("        refiner_coarse_output_convert", seconds_to_ms(refiner_coarse_detail.get("output_convert"))),
      ("        refiner_coarse_encodeA", seconds_to_ms(refiner_coarse_detail.get("encodeA"))),
      ("        refiner_coarse_encodeAB", seconds_to_ms(refiner_coarse_detail.get("encodeAB"))),
      ("        refiner_coarse_trans_head", seconds_to_ms(refiner_coarse_detail.get("trans_head"))),
      ("        refiner_coarse_rot_head", seconds_to_ms(refiner_coarse_detail.get("rot_head"))),
      ("      refiner_coarse_pose_update", seconds_to_ms(refiner_coarse_detail.get("pose_update"))),
      ("      refiner_coarse_empty_cache", seconds_to_ms(refiner_coarse_detail.get("empty_cache"))),
      ("      refiner_coarse_other", seconds_to_ms(refiner_coarse_detail.get("other"))),
      ("    refiner_fine", seconds_to_ms(timing.get("refiner_fine"))),
      ("      refiner_fine_backend", refiner_fine_detail.get("network_backend")),
      ("      refiner_fine_network_internal_sync", refiner_fine_detail.get("network_internal_sync_status")),
      ("      refiner_fine_fallback", refiner_fine_detail.get("fallback_reason")),
      ("      refiner_fine_crop_window", seconds_to_ms(refiner_fine_detail.get("crop_window"))),
      ("      refiner_fine_render", seconds_to_ms(refiner_fine_detail.get("render"))),
      ("      refiner_fine_render_postprocess", seconds_to_ms(refiner_fine_detail.get("render_postprocess"))),
      ("      refiner_fine_warp", seconds_to_ms(refiner_fine_detail.get("warp"))),
      ("      refiner_fine_transform", seconds_to_ms(refiner_fine_detail.get("transform"))),
      ("      refiner_fine_input_pack", seconds_to_ms(refiner_fine_detail.get("input_pack"))),
      ("      refiner_fine_network_forward", seconds_to_ms(refiner_fine_detail.get("network_forward"))),
      ("        refiner_fine_layout_convert", seconds_to_ms(refiner_fine_detail.get("layout_convert"))),
      ("        refiner_fine_dtype_convert", seconds_to_ms(refiner_fine_detail.get("dtype_convert"))),
      ("        refiner_fine_tensorrt_execute", seconds_to_ms(refiner_fine_detail.get("tensorrt_execute"))),
      ("        refiner_fine_output_convert", seconds_to_ms(refiner_fine_detail.get("output_convert"))),
      ("        refiner_fine_encodeA", seconds_to_ms(refiner_fine_detail.get("encodeA"))),
      ("        refiner_fine_encodeAB", seconds_to_ms(refiner_fine_detail.get("encodeAB"))),
      ("        refiner_fine_trans_head", seconds_to_ms(refiner_fine_detail.get("trans_head"))),
      ("        refiner_fine_rot_head", seconds_to_ms(refiner_fine_detail.get("rot_head"))),
      ("      refiner_fine_pose_update", seconds_to_ms(refiner_fine_detail.get("pose_update"))),
      ("      refiner_fine_empty_cache", seconds_to_ms(refiner_fine_detail.get("empty_cache"))),
      ("      refiner_fine_other", seconds_to_ms(refiner_fine_detail.get("other"))),
      ("  foundation_coarse_score_select", seconds_to_ms(timing.get("coarse_score_select"))),
      ("  foundation_coarse_scorer_status", timing.get("coarse_scorer_status")),
      ("  foundation_scorer", seconds_to_ms(timing.get("scorer"))),
      ("    scorer_coarse", seconds_to_ms(scorer_coarse_time) if scorer_coarse_ran else None),
      ("      scorer_coarse_backend", scorer_coarse_detail.get("network_backend")),
      ("      scorer_coarse_network_internal_sync", scorer_coarse_detail.get("network_internal_sync_status")),
      ("      scorer_coarse_fallback", scorer_coarse_detail.get("fallback_reason")),
      ("      scorer_coarse_crop_window", seconds_to_ms(scorer_coarse_detail.get("crop_window"))),
      ("      scorer_coarse_render", seconds_to_ms(scorer_coarse_detail.get("render"))),
      ("      scorer_coarse_render_postprocess", seconds_to_ms(scorer_coarse_detail.get("render_postprocess"))),
      ("      scorer_coarse_warp", seconds_to_ms(scorer_coarse_detail.get("warp"))),
      ("      scorer_coarse_transform", seconds_to_ms(scorer_coarse_detail.get("transform"))),
      ("      scorer_coarse_input_pack", seconds_to_ms(scorer_coarse_detail.get("input_pack"))),
      ("      scorer_coarse_network_forward", seconds_to_ms(scorer_coarse_detail.get("network_forward"))),
      ("        scorer_coarse_layout_convert", seconds_to_ms(scorer_coarse_detail.get("layout_convert"))),
      ("        scorer_coarse_dtype_convert", seconds_to_ms(scorer_coarse_detail.get("dtype_convert"))),
      ("        scorer_coarse_tensorrt_execute", seconds_to_ms(scorer_coarse_detail.get("tensorrt_execute"))),
      ("        scorer_coarse_output_convert", seconds_to_ms(scorer_coarse_detail.get("output_convert"))),
      ("        scorer_coarse_encoderA", seconds_to_ms(scorer_coarse_detail.get("encoderA"))),
      ("        scorer_coarse_encoderAB", seconds_to_ms(scorer_coarse_detail.get("encoderAB"))),
      ("        scorer_coarse_self_attention", seconds_to_ms(scorer_coarse_detail.get("self_attention"))),
      ("        scorer_coarse_cross_attention", seconds_to_ms(scorer_coarse_detail.get("cross_attention"))),
      ("        scorer_coarse_linear", seconds_to_ms(scorer_coarse_detail.get("linear"))),
      ("      scorer_coarse_empty_cache", seconds_to_ms(scorer_coarse_detail.get("empty_cache"))),
      ("      scorer_coarse_other", seconds_to_ms(scorer_coarse_detail.get("other"))),
      ("    scorer_fine", seconds_to_ms(timing.get("scorer_fine")) if scorer_coarse_ran else None),
      (f"{scorer_fine_detail_indent}scorer_fine_backend", scorer_fine_detail.get("network_backend")),
      (f"{scorer_fine_detail_indent}scorer_fine_network_internal_sync", scorer_fine_detail.get("network_internal_sync_status")),
      (f"{scorer_fine_detail_indent}scorer_fine_fallback", scorer_fine_detail.get("fallback_reason")),
      (f"{scorer_fine_detail_indent}scorer_fine_crop_window", seconds_to_ms(scorer_fine_detail.get("crop_window"))),
      (f"{scorer_fine_detail_indent}scorer_fine_render", seconds_to_ms(scorer_fine_detail.get("render"))),
      (f"{scorer_fine_detail_indent}scorer_fine_render_postprocess", seconds_to_ms(scorer_fine_detail.get("render_postprocess"))),
      (f"{scorer_fine_detail_indent}scorer_fine_warp", seconds_to_ms(scorer_fine_detail.get("warp"))),
      (f"{scorer_fine_detail_indent}scorer_fine_transform", seconds_to_ms(scorer_fine_detail.get("transform"))),
      (f"{scorer_fine_detail_indent}scorer_fine_input_pack", seconds_to_ms(scorer_fine_detail.get("input_pack"))),
      (f"{scorer_fine_detail_indent}scorer_fine_network_forward", seconds_to_ms(scorer_fine_detail.get("network_forward"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_layout_convert", seconds_to_ms(scorer_fine_detail.get("layout_convert"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_dtype_convert", seconds_to_ms(scorer_fine_detail.get("dtype_convert"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_tensorrt_execute", seconds_to_ms(scorer_fine_detail.get("tensorrt_execute"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_output_convert", seconds_to_ms(scorer_fine_detail.get("output_convert"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_encoderA", seconds_to_ms(scorer_fine_detail.get("encoderA"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_encoderAB", seconds_to_ms(scorer_fine_detail.get("encoderAB"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_self_attention", seconds_to_ms(scorer_fine_detail.get("self_attention"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_cross_attention", seconds_to_ms(scorer_fine_detail.get("cross_attention"))),
      (f"{scorer_fine_detail_indent}  scorer_fine_linear", seconds_to_ms(scorer_fine_detail.get("linear"))),
      (f"{scorer_fine_detail_indent}scorer_fine_empty_cache", seconds_to_ms(scorer_fine_detail.get("empty_cache"))),
      (f"{scorer_fine_detail_indent}scorer_fine_other", seconds_to_ms(scorer_fine_detail.get("other"))),
      ("  foundation_topk_select", seconds_to_ms(timing.get("topk_select"))),
      ("  foundation_sort_select", seconds_to_ms(timing.get("sort_select"))),
      ("  foundation_other", seconds_to_ms(timing.get("other"))),
  ]

  detail_by_prefix = {
      "refiner_coarse": refiner_coarse_detail,
      "refiner_fine": refiner_fine_detail,
      "scorer_coarse": scorer_coarse_detail,
      "scorer_fine": scorer_fine_detail,
  }
  network_detail_suffixes = {
      "layout_convert",
      "dtype_convert",
      "tensorrt_execute",
      "output_convert",
      "encodeA",
      "encodeAB",
      "trans_head",
      "rot_head",
      "encoderA",
      "encoderAB",
      "self_attention",
      "cross_attention",
      "linear",
  }
  render_profile_keys = (
      "render_context_mesh_check",
      "render_projection_setup",
      "render_vertex_camera_transform",
      "render_vertex_homogeneous",
      "render_vertex_clip_transform",
      "render_bbox_transform",
      "render_rasterize",
      "render_xyz_depth_interpolate",
      "render_texture_sample",
      "render_normal_transform",
      "render_normal_interpolate",
      "render_normal_normalize_flip",
      "render_diffuse_vertex",
      "render_diffuse_interpolate",
      "render_lighting_blend",
      "render_finalize_flip_mask",
      "render_profile_other",
  )
  render_profile_prefixes = {
      "refiner_coarse",
      "refiner_fine",
      "scorer_coarse",
      "scorer_fine",
  }
  transform_profile_keys = (
      "transform_xyzB_precomputed_warp_crop",
      "transform_batch_setup",
      "transform_rgb_normalize",
      "transform_xyzA_normalize_mask",
      "transform_xyzB_unwarp_depth",
      "transform_xyzB_depth_to_xyz",
      "transform_xyzB_warp_crop",
      "transform_xyzB_normalize_mask",
      "transform_profile_other",
  )
  transform_profile_prefixes = {
      "scorer_coarse",
      "scorer_fine",
  }
  filtered_rows = []
  for stage, value in rows:
    normalized_stage = stage.strip()
    matched_prefix = next(
        (
            prefix
            for prefix in detail_by_prefix
            if normalized_stage.startswith(f"{prefix}_")
        ),
        None,
    )
    if matched_prefix is not None:
      suffix = normalized_stage[len(matched_prefix) + 1:]
      detail = detail_by_prefix[matched_prefix]
      backend = detail.get("network_backend")
      if suffix == "backend":
        continue
      if suffix == "network_internal_sync" and value == "disabled":
        continue
      if suffix == "empty_cache" and (value is None or float(value) == 0.0):
        continue
      if suffix == "fallback" and not value:
        continue
      if suffix == "render" and matched_prefix in render_profile_prefixes:
        if value is not None:
          filtered_rows.append((stage, value))
        child_indent = stage[:len(stage) - len(stage.lstrip())] + "  "
        for profile_key in render_profile_keys:
          profile_value = seconds_to_ms(detail.get(profile_key))
          if profile_value is not None:
            filtered_rows.append(
                (f"{child_indent}{matched_prefix}_{profile_key}", profile_value)
            )
        continue
      if suffix == "transform" and matched_prefix in transform_profile_prefixes:
        if value is not None:
          filtered_rows.append((stage, value))
        child_indent = stage[:len(stage) - len(stage.lstrip())] + "  "
        for profile_key in transform_profile_keys:
          profile_value = seconds_to_ms(detail.get(profile_key))
          if profile_value is not None:
            filtered_rows.append(
                (f"{child_indent}{matched_prefix}_{profile_key}", profile_value)
            )
        continue
      if backend == "tensorrt":
        if suffix == "network_forward":
          output_convert_ms = seconds_to_ms(detail.get("output_convert")) or 0.0
          value = (value or 0.0) + output_convert_ms
          stage = stage.replace(
              f"{matched_prefix}_network_forward",
              f"{matched_prefix}_network_total(T)",
          )
        elif suffix in network_detail_suffixes:
          continue
    if value is not None:
      filtered_rows.append((stage, value))
  return filtered_rows


def foundation_candidate_stage_rows(timing: dict) -> list[tuple[str, int | None, float | None]]:
  if "refiner_coarse_candidates" in timing:
    rows = [
        ("pose_hypothesis_generated", timing.get("pose_hypothesis_candidates"), timing.get("pose_hypothesis")),
        ("axis_prior_after", timing.get("axis_prior_candidates_after"), timing.get("axis_prior")),
        ("refiner_coarse_input", timing.get("refiner_coarse_candidates"), timing.get("refiner_coarse")),
    ]
    if float(timing.get("scorer_coarse", 0.0) or 0.0) > 0.0:
      rows.append(
          ("scorer_coarse_input", timing.get("scorer_coarse_candidates"), timing.get("scorer_coarse"))
      )
    rows.extend(
        (
            ("refiner_fine_input", timing.get("refiner_fine_candidates"), timing.get("refiner_fine")),
            ("scorer_fine_input", timing.get("scorer_fine_candidates"), timing.get("scorer_fine")),
        )
    )
    return rows
  return [
      ("pose_hypothesis_generated", timing.get("pose_hypothesis_candidates"), timing.get("pose_hypothesis")),
      ("axis_prior_after", timing.get("axis_prior_candidates_after"), timing.get("axis_prior")),
      ("refiner_input", timing.get("refiner_candidates"), timing.get("refiner")),
      ("scorer_input", timing.get("scorer_candidates"), timing.get("scorer")),
  ]


def print_foundation_candidate_summary(init_index: int, timing: dict) -> None:
  status = timing.get("axis_prior_status", "unknown")
  confidence = timing.get("axis_prior_confidence")
  confidence_text = "N/A" if confidence is None else f"{float(confidence):.3f}"
  confidence_threshold = timing.get("axis_prior_min_confidence")
  confidence_threshold_text = "N/A" if confidence_threshold is None else f"{float(confidence_threshold):.3f}"
  confidence_passed = (
      confidence is not None
      and confidence_threshold is not None
      and np.isfinite(float(confidence))
      and float(confidence) >= float(confidence_threshold)
  )
  eigenvalues = timing.get("axis_prior_eigenvalues")
  eigenvalues_text = "N/A" if eigenvalues is None else "[" + ", ".join(f"{float(value):.6g}" for value in eigenvalues) + "]"
  print(
      f"[PCA][SUMMARY][foundationpose_init] init={init_index} "
      f"status={status} filter={timing.get('axis_prior_filter', 'unknown')} "
      f"points={int(timing.get('axis_prior_points', 0))} "
      f"confidence={confidence_text}/{confidence_threshold_text} passed={confidence_passed} "
      f"eigenvalues={eigenvalues_text} "
      f"model_axis={timing.get('axis_prior_model_axis')} max_angle_deg={timing.get('axis_prior_max_angle_deg')} "
      f"min_candidates={timing.get('axis_prior_min_candidates')} "
      f"max_candidates={timing.get('axis_prior_max_candidates_config')}"
  )

  stage_width = 31
  count_width = 10
  total_width = 14
  unit_width = 18
  line_width = stage_width + count_width + total_width + unit_width
  print(f"[CANDIDATES][SUMMARY][foundationpose_init] init={init_index}")
  print(
      f"{'stage':<{stage_width}}"
      f"{'count':>{count_width}}"
      f"{'total_ms':>{total_width}}"
      f"{'ms_per_candidate':>{unit_width}}"
  )
  print("-" * line_width)
  for stage, count, seconds in foundation_candidate_stage_rows(timing):
    normalized_count = None if count is None else int(count)
    total_ms = seconds_to_ms(seconds)
    if normalized_count is None or total_ms is None:
      continue
    per_candidate_ms = None
    if total_ms is not None and normalized_count is not None and normalized_count > 0:
      per_candidate_ms = total_ms / normalized_count
    count_text = "N/A" if normalized_count is None else str(normalized_count)
    total_text = "N/A" if total_ms is None else f"{total_ms:.3f}"
    unit_text = "N/A" if per_candidate_ms is None else f"{per_candidate_ms:.3f}"
    print(
        f"{stage:<{stage_width}}"
        f"{count_text:>{count_width}}"
        f"{total_text:>{total_width}}"
        f"{unit_text:>{unit_width}}"
    )


def print_timing_summary(init_index: int, rows: list[tuple[str, float | str | None]]) -> None:
  stage_width = 58 if any("_render_context_mesh_check" in stage for stage, _ in rows) else 44
  value_width = 12
  line_width = stage_width + value_width + 1
  print(f"[TIMER][SUMMARY][foundationpose_init] init={init_index} unit=ms")
  print(f"{'stage':<{stage_width}}{'time_ms':>{value_width}}")
  print("-" * line_width)
  for stage, value in rows:
    if stage == "":
      print()
      continue
    if set(stage) == {"-"}:
      print(stage)
      continue
    if value is None:
      continue
    if isinstance(value, str):
      value_text = value
    else:
      value_text = f"{float(value):.3f}"
    print(f"{stage:<{stage_width}}{value_text:>{value_width}}")


def log_runtime(enabled: bool, message: str) -> None:
  if enabled:
    print(message)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
  parser.add_argument("--camera-source", choices=("realsense", "ros2"), default=None)
  args = parser.parse_args()
  cfg = load_config(args.config)

  camera_cfg = cfg.get("camera", {})
  yolo_cfg = cfg.get("yolo", {})
  tracker_cfg = cfg.get("foundationpose", {})
  runtime_cfg = cfg.get("runtime", {})
  runtime_mode = str(runtime_cfg.get("mode", "realtime")).lower()
  init_only = runtime_mode in ("init_only", "initialize_only", "register_only")
  success_timing_only = bool(runtime_cfg.get("success_timing_only", False))
  verbose_runtime = not success_timing_only

  camera = build_camera_reader(
      camera_cfg,
      config_path=args.config,
      source_override=args.camera_source,
  )
  if hasattr(camera, "apply_model_cpu_affinity"):
    camera.apply_model_cpu_affinity()
  yolo_imgsz = yolo_cfg.get("imgsz", 640)
  detector = YoloSegmenter(
      weights=resolve_path(yolo_cfg["weights"]),
      target_class=yolo_cfg.get("target_class"),
      target_class_id=yolo_cfg.get("target_class_id"),
      conf=float(yolo_cfg.get("conf", 0.35)),
      imgsz=yolo_imgsz,
      device=yolo_cfg.get("device"),
      half=bool(yolo_cfg.get("half", True)),
      min_mask_area=int(yolo_cfg.get("min_mask_area", 100)),
      morph_kernel=int(yolo_cfg.get("morph_kernel", 5)),
      execution_path=str(yolo_cfg.get("execution_path", "legacy")),
      profile_stages=bool(yolo_cfg.get("profile_stages", False)),
      fallback_to_legacy=bool(yolo_cfg.get("fallback_to_legacy", True)),
      postprocess_backend=str(yolo_cfg.get("postprocess_backend", "gpu")),
  )
  tracker = build_tracker(tracker_cfg)
  if success_timing_only:
    logging.getLogger().setLevel(logging.WARNING)

  detection_interval = int(runtime_cfg.get("detection_interval", 15))
  relocalize_iou = float(runtime_cfg.get("relocalize_iou", 0.25))
  min_valid_depth_pixels = int(runtime_cfg.get("min_valid_depth_pixels", 500))
  max_missing_detections = int(runtime_cfg.get("max_missing_detections", 3))
  status_log_interval = int(runtime_cfg.get("status_log_interval", 30))
  register_retry_interval = int(runtime_cfg.get("register_retry_interval", 30))
  register_required_stable_detections = int(runtime_cfg.get("register_required_stable_detections", 1))
  register_max_area_change = float(runtime_cfg.get("register_max_area_change", 0.0))
  register_border_margin = int(runtime_cfg.get("register_border_margin", 0))
  show_window = bool(runtime_cfg.get("show_window", True))
  pose_output = resolve_path(runtime_cfg.get("pose_output", "realtime_foundation/outputs/latest_pose.txt"))
  recording_cfg = cfg.get("recording", {})
  record_writer = None
  record_save_register = bool(recording_cfg.get("save_register", True))
  record_save_track = bool(recording_cfg.get("save_track", True))
  record_require_same_frame_mask = bool(recording_cfg.get("require_same_frame_mask", True))
  record_save_topk_poses = bool(recording_cfg.get("save_topk_poses", False))
  record_topk_pose_count = int(recording_cfg.get("topk_pose_count", 5))
  if bool(recording_cfg.get("enabled", False)):
    record_writer = PoseRecordWriter(
        output_dir=resolve_path(recording_cfg.get("output_dir", "realtime_foundation/outputs/frame_records")),
        jpeg_quality=int(recording_cfg.get("jpeg_quality", 90)),
        config=cfg,
        reset_on_start=bool(recording_cfg.get("reset_on_start", True)),
    )
    record_writer.start()

  frame_index = 0
  last_processed_frame_id = 0
  last_processed_detection_sequence = 0
  last_tracking_detection_frame_id = 0
  last_tracking_detection_sequence = 0
  init_count = 0
  last_success_time = None
  last_detection = None
  current_frame_detection = None
  missing_detection_count = 0
  register_stable_areas = []
  next_register_frame = 0
  frame_publisher = RgbdImagePublisher(camera)
  sync_recorded_tracking = record_writer is not None and record_save_track and record_require_same_frame_mask
  default_yolo_worker_frame_stride = 1 if init_only or sync_recorded_tracking else max(1, detection_interval)
  yolo_worker = YoloDetectionWorker(
      detector=detector,
      image_topic=frame_publisher.image_topic,
      frame_stride=int(runtime_cfg.get("yolo_worker_frame_stride", default_yolo_worker_frame_stride)),
  )

  def request_termination(signum, _frame):
    raise TerminationRequested(f"received signal {signum}")

  previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
  signal.signal(signal.SIGTERM, request_termination)
  try:
    frame_publisher.start()
    yolo_worker.start()
    while True:
      reset_tracker_after_record = False
      current_frame_detection = None
      quality_rendered_mask = None
      initializing = not tracker.initialized

      if initializing:
        detection_snapshot = yolo_worker.wait_for_detection(
            last_processed_detection_sequence,
            timeout=0.1,
        )
        if detection_snapshot is None:
          # Surface camera-thread failures while waiting for YOLO. Normal
          # delivery wakes the condition immediately; this timeout is only a
          # health check and does not gate detection on another camera frame.
          frame_publisher.get_latest()
          continue
        detection_sequence, detection_message = detection_snapshot
        last_processed_detection_sequence = detection_sequence
        frame_index = detection_message.frame_id
        last_processed_frame_id = frame_index
        color, depth, K = detection_message.color, detection_message.depth, detection_message.K
        processed_timestamp = detection_message.timestamp
        detection = detection_message.detection
        yolo_time = detection_message.yolo_time
        yolo_timing = detection_message.yolo_timing
        yolo_candidate_count = detection_message.yolo_candidate_count
        detection_consumed_unix_ns = time.time_ns()
        detection_consumed_monotonic_ns = time.perf_counter_ns()
        init_start_monotonic_ns = detection_message.yolo_start_monotonic_ns
        current_frame_detection = detection
        if detection is None:
          register_stable_areas.clear()
          if should_log_status(frame_index, status_log_interval):
            log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: waiting for YOLO target mask")
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          continue

        if frame_index < next_register_frame:
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          yolo_worker.resume_candidate_search(detection_sequence)
          continue

      else:
        frame_message = frame_publisher.get_latest()
        if frame_message is None or frame_message.frame_id == last_processed_frame_id:
          time.sleep(0.001)
          continue
        last_processed_frame_id = frame_message.frame_id
        color, depth, K = frame_message.color, frame_message.depth, frame_message.K
        frame_index = frame_message.frame_id
        processed_timestamp = frame_message.timestamp

      if initializing:
        validation_start = time.perf_counter()
        valid_depth_pixels = count_valid_depth_pixels(depth, detection.mask)
        log_runtime(
            verbose_runtime,
            f"[Realtime] Frame {frame_index}: YOLO target class={detection.class_name} "
            f"conf={detection.confidence:.3f}, mask_area={detection.area}, valid_depth={valid_depth_pixels}",
        )
        reject_reason = detection_reject_reason(detection, depth, runtime_cfg)
        if reject_reason is not None:
          register_stable_areas.clear()
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: reject YOLO detection: {reject_reason}")
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          yolo_worker.resume_candidate_search(detection_sequence)
          continue
        if valid_depth_pixels < min_valid_depth_pixels:
          register_stable_areas.clear()
          log_runtime(
              verbose_runtime,
              f"[Realtime] Frame {frame_index}: valid depth inside mask is too small "
              f"({valid_depth_pixels} < {min_valid_depth_pixels}); waiting",
          )
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          yolo_worker.resume_candidate_search(detection_sequence)
          continue

        border_reject_reason = detection_border_reject_reason(detection, color.shape[:2], register_border_margin)
        if border_reject_reason is not None:
          register_stable_areas.clear()
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: wait for full target before register: {border_reject_reason}")
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          yolo_worker.resume_candidate_search(detection_sequence)
          continue

        stability_reject_reason = register_stability_reject_reason(
            detection,
            register_stable_areas,
            register_required_stable_detections,
            register_max_area_change,
        )
        if stability_reject_reason is not None:
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: wait for stable YOLO mask before register: {stability_reject_reason}")
          if show_window:
            cv2.imshow("realtime_foundation", color[..., ::-1])
            if cv2.waitKey(1) in (27, ord("q")):
              break
          yolo_worker.resume_candidate_search(detection_sequence)
          continue

        last_detection = detection
        missing_detection_count = 0
        register_stable_areas.clear()
        validation_time = time.perf_counter() - validation_start
        log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: registering FoundationPose")
        # Keep YOLO off the GPU for the complete registration GPU phase. In
        # particular, registration-quality rendering must finish before the
        # detection worker is resumed, otherwise both workloads contend on
        # the single Orin GPU and inflate each other's latency.
        yolo_worker.pause()
        try:
          try:
            register_start_unix_ns = time.time_ns()
            register_start = time.perf_counter()
            pose_result = tracker.register(color, depth, K, detection.mask)
            register_wall_time = time.perf_counter() - register_start
            register_end_unix_ns = time.time_ns()
          except Exception as exc:
            tracker.reset()
            next_register_frame = frame_index + register_retry_interval
            log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: register failed, waiting for next detection: {exc}")
            log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: retry register after frame {next_register_frame}")
            yolo_worker.resume_candidate_search(detection_sequence)
            continue
          timing = getattr(tracker.estimator, "last_register_timing", {})
          if record_writer is not None and record_save_topk_poses:
            try:
              topk_record = build_topk_pose_record(
                  tracker=tracker,
                  color=color,
                  K=K,
                  target_mask=detection.mask,
                  frame_id=frame_index,
                  count=record_topk_pose_count,
              )
              if topk_record is not None:
                record_writer.submit(topk_record)
            except Exception as exc:
              print(f"[TopK] Frame {frame_index}: failed to build ranked pose diagnostics: {exc}")
          quality_result = evaluate_register_quality(
              tracker=tracker,
              pose=pose_result.pose,
              depth=depth,
              K=K,
              mask=detection.mask,
              runtime_cfg=runtime_cfg,
            )
        finally:
          yolo_worker.resume()
        if tracker.quality_render_mask_reuse_enabled:
          quality_rendered_mask = quality_result.rendered_mask
        if not quality_result.accepted:
          log_runtime(
              verbose_runtime,
              f"[Realtime] Frame {frame_index}: reject initialization: {quality_result.reject_reason} "
              f"translation_drift={quality_result.translation_drift} "
              f"render_iou={quality_result.render_iou} "
              f"score_gap={quality_result.top1_top2_score_gap}",
          )
          tracker.reset()
          yolo_worker.resume_candidate_search(detection_sequence)
          continue
        if not init_only:
          yolo_worker.finish_candidate_search(detection_sequence)
        init_count += 1
        last_tracking_detection_frame_id = frame_index
        last_tracking_detection_sequence = detection_sequence
        init_success_monotonic_ns = time.perf_counter_ns()
        init_success_unix_ns = time.time_ns()
        init_total_time = (init_success_monotonic_ns - init_start_monotonic_ns) / 1_000_000_000.0
        detection_consume_delay = max(
          0.0,
          (detection_consumed_monotonic_ns - detection_message.yolo_end_monotonic_ns)
          / 1_000_000_000.0,
        )
        yolo_total_value = yolo_timing.get("total")
        yolo_total_time = float(yolo_time if yolo_total_value is None else yolo_total_value)
        foundation_total_value = timing.get("register")
        foundation_total_time = float(
          register_wall_time if foundation_total_value is None else foundation_total_value
        )
        yolo_foundation_compute_total_time = yolo_total_time + foundation_total_time
        success_time = time.perf_counter()
        success_period = None if last_success_time is None else success_time - last_success_time
        last_success_time = success_time
        success_hz = None if success_period is None else 1.0 / max(success_period, 1e-6)
        log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: FoundationPose initialized")
        print(
          f"[EVENT][foundationpose_init] schema=1 init={init_count} frame={frame_index} "
          f"camera_timestamp={processed_timestamp:.9f} "
          f"yolo_start_unix_ns={detection_message.yolo_start_unix_ns} "
          f"yolo_end_unix_ns={detection_message.yolo_end_unix_ns} "
          f"detection_consumed_unix_ns={detection_consumed_unix_ns} "
          f"register_start_unix_ns={register_start_unix_ns} "
          f"register_end_unix_ns={register_end_unix_ns} "
          f"init_success_unix_ns={init_success_unix_ns}"
        )
        print_foundation_candidate_summary(init_count, timing)
        print_timing_summary(
            init_count,
            [
              ("yolo_total", seconds_to_ms(yolo_total_time)),
                ("yolo_candidate_count", str(yolo_candidate_count)),
                ("  yolo_model_predict", seconds_to_ms(yolo_timing.get("model_predict"))),
              ("    yolo_preprocess", seconds_to_ms(yolo_timing.get("model_preprocess"))),
              ("    yolo_inference", seconds_to_ms(yolo_timing.get("model_inference"))),
              ("    yolo_postprocess_internal", seconds_to_ms(yolo_timing.get("model_postprocess"))),
              ("    yolo_framework_overhead", seconds_to_ms(yolo_timing.get("model_framework_overhead"))),
                ("  yolo_tensor_to_cpu", seconds_to_ms(yolo_timing.get("tensor_to_cpu"))),
                ("  yolo_mask_resize_clean", seconds_to_ms(yolo_timing.get("mask_resize_clean"))),
                ("  yolo_select_best_mask", seconds_to_ms(yolo_timing.get("select_best_mask"))),
                ("  detection_consume_delay", seconds_to_ms(detection_consume_delay)),
                ("", None),
                *foundation_register_timing_rows(timing, register_wall_time),
                ("", None),
                ("yolo_foundation_compute_total", seconds_to_ms(yolo_foundation_compute_total_time)),
                ("init_total", seconds_to_ms(init_total_time)),
                ("success_period", seconds_to_ms(success_period)),
                ("-" * 57, None),
                ("success_rate_hz", success_hz),
            ],
        )
        try:
          tracker.save_render_lod_comparison(
              color=color,
              K=K,
              frame_id=frame_index,
          )
        except Exception as exc:
          print(f"[RenderLOD] Frame {frame_index}: failed to save runtime comparison: {exc}")
      else:
        if sync_recorded_tracking:
          detection_sequence, detection_message = yolo_worker.get_latest_snapshot()
          if detection_message is None or detection_sequence <= last_tracking_detection_sequence:
            continue
          last_tracking_detection_sequence = detection_sequence
          last_tracking_detection_frame_id = detection_message.frame_id
          detection = detection_message.detection
          if detection is None:
            missing_detection_count += 1
            if missing_detection_count >= max_missing_detections:
              yolo_worker.pause()
              try:
                tracker.reset()
                last_detection = None
                missing_detection_count = 0
                register_stable_areas.clear()
                last_processed_detection_sequence = yolo_worker.enable_candidate_search()
              finally:
                yolo_worker.resume()
            continue
          missing_detection_count = 0
          last_detection = detection
          frame_index = detection_message.frame_id
          processed_timestamp = detection_message.timestamp
          color, depth, K = detection_message.color, detection_message.depth, detection_message.K
          current_frame_detection = detection
        try:
          pose_result = tracker.track(color, depth, K)
        except Exception as exc:
          yolo_worker.pause()
          try:
            tracker.reset()
            last_detection = None
            missing_detection_count = 0
            register_stable_areas.clear()
            last_processed_detection_sequence = yolo_worker.enable_candidate_search()
          finally:
            yolo_worker.resume()
          log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: tracking failed, waiting for detection: {exc}")
          continue
        if not sync_recorded_tracking and should_check_detection(frame_index, detection_interval):
          detection_sequence, detection_message = yolo_worker.get_latest_snapshot()
          if detection_message is None or detection_sequence <= last_tracking_detection_sequence:
            current_frame_detection = None
            log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: waiting for a new YOLO detection message during tracking")
          else:
            last_tracking_detection_sequence = detection_sequence
            last_tracking_detection_frame_id = detection_message.frame_id
            detection = detection_message.detection
            current_frame_detection = detection if detection_message.frame_id == frame_index else None
            if detection is None:
              missing_detection_count += 1
              log_runtime(
                  verbose_runtime,
                  f"[Realtime] Frame {frame_index}: YOLO target missing during tracking "
                  f"({missing_detection_count}/{max_missing_detections})",
              )
              if missing_detection_count >= max_missing_detections:
                last_detection = None
                log_runtime(verbose_runtime, f"[Realtime] Frame {frame_index}: target lost, stop FoundationPose tracking and wait for YOLO detection")
                reset_tracker_after_record = True
            else:
              missing_detection_count = 0
              last_detection = detection
              log_runtime(
                  verbose_runtime,
                  f"[Realtime] Frame {frame_index}: YOLO target present during tracking "
                  f"class={detection.class_name} conf={detection.confidence:.3f}, keep FoundationPose tracking",
              )

      save_pose(pose_output, pose_result.pose)

      should_record_pose = record_writer is not None and (
          (pose_result.mode == "register" and record_save_register)
          or (pose_result.mode == "track" and record_save_track)
        ) and current_frame_detection is not None
      if show_window or should_record_pose:
        vis = tracker.draw_visualization(
            color,
            K,
            pose_result.pose,
            rendered_mask=quality_rendered_mask,
        )
        if current_frame_detection is not None:
          mask_overlay = current_frame_detection.mask.astype(bool)
          vis[mask_overlay] = (0.65 * vis[mask_overlay] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
        if should_record_pose:
          confidence = None if current_frame_detection is None else float(current_frame_detection.confidence)
          record_vis = add_pose_record_label(vis, frame_index, pose_result.mode, confidence)
          filename_stem = f"frame_{frame_index:06d}_{pose_result.mode}"
          record_writer.submit(PoseRecord(
              filename_stem=filename_stem,
              visualization=record_vis.copy(),
              rgb=np.asarray(color, dtype=np.uint8).copy(),
              depth=np.asarray(depth, dtype=np.float32).copy(),
              K=np.asarray(K, dtype=np.float32).copy(),
              mode=pose_result.mode,
              frame_id=frame_index,
              timestamp=processed_timestamp,
              yolo_mask=np.asarray(current_frame_detection.mask, dtype=np.uint8).copy(),
              yolo_confidence=float(confidence) if confidence is not None else float("nan"),
              output_pose=np.asarray(pose_result.pose, dtype=np.float32).reshape(4, 4).copy(),
          ))
        if show_window:
          cv2.imshow("realtime_foundation", vis[..., ::-1])
          if cv2.waitKey(1) in (27, ord("q")):
            break

      if init_only and pose_result.mode == "register":
        tracker.reset()
        last_detection = None
        missing_detection_count = 0
        register_stable_areas.clear()
        yolo_worker.resume_candidate_search(detection_sequence)
        continue

      if reset_tracker_after_record:
        yolo_worker.pause()
        try:
          tracker.reset()
          missing_detection_count = 0
          register_stable_areas.clear()
          last_processed_detection_sequence = yolo_worker.enable_candidate_search()
        finally:
          yolo_worker.resume()
        continue

      time.sleep(float(runtime_cfg.get("loop_sleep", 0.0)))
  except TerminationRequested as exc:
    print(f"[Realtime] {exc}; stopping")
  finally:
    yolo_worker.stop()
    frame_publisher.stop()
    if record_writer is not None:
      record_writer.stop()
    cv2.destroyAllWindows()
    signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
  main()