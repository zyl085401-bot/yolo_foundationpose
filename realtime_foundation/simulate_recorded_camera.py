from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
import shutil
import time

import cv2
import numpy as np

from run_realtime import (
    FrameMessage,
    LatestTopic,
    MaskMessage,
    add_pose_record_label,
    build_topk_pose_record,
    build_tracker,
    count_valid_depth_pixels,
    detection_border_reject_reason,
    detection_reject_reason,
    evaluate_register_quality,
    foundation_register_timing_rows,
    load_config,
    print_foundation_candidate_summary,
    print_timing_summary,
    register_stability_reject_reason,
    seconds_to_ms,
)


@dataclass(frozen=True)
class ReplayDetection:
  mask: np.ndarray
  confidence: float
  area: int
  box_xyxy: tuple[float, float, float, float]


class RecordedFramePublisher:
  def __init__(self, input_path: str):
    self.input_path = os.path.abspath(input_path)
    self.image_topic = LatestTopic("/sim_camera/rgbd/latest")
    self.mask_topic = LatestTopic("/sim_camera/mask/latest")
    with np.load(self.input_path, allow_pickle=False) as data:
      self.frame_id = int(data["frame_id"].item())
      self.timestamp = float(data["timestamp"].item())
      self.rgb = np.asarray(data["rgb"], dtype=np.uint8)
      self.depth = np.asarray(data["depth"], dtype=np.float32)
      self.K = np.asarray(data["K"], dtype=np.float32)
      self.mask = np.asarray(data["yolo_mask"], dtype=np.uint8)
      self.confidence = float(data["yolo_confidence"].item())
      self.saved_output_pose = np.asarray(data["output_pose"], dtype=np.float32).reshape(4, 4)
    if self.mask.shape != self.depth.shape[:2]:
      raise ValueError(f"mask/depth shape mismatch: mask={self.mask.shape}, depth={self.depth.shape}")

  def detection(self) -> ReplayDetection:
    ys, xs = np.where(self.mask > 0)
    if len(xs) == 0:
      box = (0.0, 0.0, 0.0, 0.0)
    else:
      box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    return ReplayDetection(
        mask=self.mask,
        confidence=self.confidence,
        area=int((self.mask > 0).sum()),
        box_xyxy=box,
    )

  def publish(self, simulated_frame_id: int) -> None:
    timestamp = self.timestamp
    self.image_topic.publish(FrameMessage(
        frame_id=simulated_frame_id,
        timestamp=timestamp,
        color=self.rgb.copy(),
        depth=self.depth.copy(),
        K=self.K.copy(),
    ))
    self.mask_topic.publish(MaskMessage(
        frame_id=simulated_frame_id,
        timestamp=timestamp,
        mask=self.mask.copy(),
        confidence=self.confidence,
    ))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Publish one recorded RGB-D/mask frame and run fresh FoundationPose registration.")
  parser.add_argument("--input", required=True, help="Path to frame_data/frame_xxxxxx_register.npz or frame_xxxxxx_track.npz.")
  parser.add_argument("--config", default=None, help="Config YAML. Defaults to the current realtime_foundation/config.yaml.")
  parser.add_argument("--output-dir", default=None, help="Defaults to frame_records/simulated_results.")
  parser.add_argument("--repeat", type=int, default=1, help="Publish and register the same sensor frame repeatedly.")
  return parser.parse_args()


def infer_paths(input_path: str, config_path: str | None, output_dir: str | None) -> tuple[str, str]:
  record_root = os.path.dirname(os.path.dirname(os.path.abspath(input_path)))
  if config_path is None:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
  output_dir = output_dir or os.path.join(record_root, "simulated_results")
  return os.path.abspath(config_path), os.path.abspath(output_dir)


def rotation_error_deg(reference_pose: np.ndarray, new_pose: np.ndarray) -> float:
  relative = new_pose[:3, :3] @ reference_pose[:3, :3].T
  cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
  return math.degrees(math.acos(cosine))


def receive_synchronized_input(publisher: RecordedFramePublisher) -> tuple[FrameMessage, MaskMessage]:
  frame = publisher.image_topic.get_latest()
  mask = publisher.mask_topic.get_latest()
  if frame is None or mask is None:
    raise RuntimeError("simulated camera topics did not publish both RGB-D and mask")
  if frame.frame_id != mask.frame_id:
    raise RuntimeError(f"simulated topic frame mismatch: rgbd={frame.frame_id}, mask={mask.frame_id}")
  return frame, mask


def save_topk_repeat(
    result_dir: str,
    repeat_stem: str,
    topk_record,
    jpeg_quality: int,
    repeat_index: int,
    source_frame_id: int,
    simulated_frame_id: int,
) -> str:
  topk_dir = os.path.join(result_dir, "top5_poses")
  os.makedirs(topk_dir, exist_ok=True)
  axis_tag = "axis_on" if topk_record.summary.get("axis_prior_enabled") else "axis_off"
  filename_stem = f"{repeat_stem}_{axis_tag}"

  for rank, (visualization, pose) in enumerate(zip(topk_record.visualizations, topk_record.output_poses), start=1):
    rank_stem = f"{filename_stem}_rank_{rank:02d}"
    image_path = os.path.join(topk_dir, f"{rank_stem}.jpg")
    if not cv2.imwrite(
        image_path,
        visualization[..., ::-1],
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    ):
      raise RuntimeError(f"failed to save simulated top-k result: {image_path}")
    np.savetxt(os.path.join(topk_dir, f"{rank_stem}_pose.txt"), pose)

  contact_path = os.path.join(topk_dir, f"{filename_stem}_contact_sheet.jpg")
  if not cv2.imwrite(
      contact_path,
      topk_record.contact_sheet[..., ::-1],
      [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
  ):
    raise RuntimeError(f"failed to save simulated top-k contact sheet: {contact_path}")

  summary = dict(topk_record.summary)
  summary.update({
      "repeat_index": repeat_index,
      "source_frame_id": source_frame_id,
      "simulated_frame_id": simulated_frame_id,
  })
  summary_path = os.path.join(topk_dir, f"{filename_stem}_summary.json")
  with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=2, ensure_ascii=False)
  return contact_path


def replay_validation_reject_reason(
    detection: ReplayDetection,
    depth: np.ndarray,
    image_shape: tuple[int, int],
    runtime_cfg: dict,
    stable_areas: list[int],
) -> tuple[str | None, int]:
  reject_reason = detection_reject_reason(detection, depth, runtime_cfg)
  valid_depth_pixels = count_valid_depth_pixels(depth, detection.mask)
  if reject_reason is None and valid_depth_pixels < int(runtime_cfg.get("min_valid_depth_pixels", 500)):
    reject_reason = (
        f"valid depth inside mask is too small "
        f"({valid_depth_pixels} < {int(runtime_cfg.get('min_valid_depth_pixels', 500))})"
    )
  if reject_reason is None:
    reject_reason = detection_border_reject_reason(
        detection,
        image_shape,
        int(runtime_cfg.get("register_border_margin", 0)),
    )
  if reject_reason is None:
    reject_reason = register_stability_reject_reason(
        detection,
        stable_areas,
        int(runtime_cfg.get("register_required_stable_detections", 1)),
        float(runtime_cfg.get("register_max_area_change", 0.0)),
    )
  return reject_reason, valid_depth_pixels


def serializable_foundation_timing(timing: dict) -> dict:
  def convert(value):
    if isinstance(value, dict):
      return {str(key): convert(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
      return [convert(item) for item in value]
    if isinstance(value, np.ndarray):
      return convert(value.tolist())
    if isinstance(value, np.generic):
      return convert(value.item())
    if isinstance(value, float) and not math.isfinite(value):
      return None
    if isinstance(value, (str, int, float, bool)) or value is None:
      return value
    return str(value)

  return convert(timing)


def candidate_rotation_update_deg(parent_pose: np.ndarray, pose: np.ndarray) -> float:
  relative = pose[:3, :3] @ parent_pose[:3, :3].T
  cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
  return math.degrees(math.acos(cosine))


def candidate_contact_sheet(images: list[np.ndarray], columns: int = 4) -> np.ndarray:
  if not images:
    return np.zeros((1, 1, 3), dtype=np.uint8)
  thumbnail_width = 320
  thumbnail_height = 240
  rows = int(math.ceil(len(images) / columns))
  sheet = np.zeros((rows * thumbnail_height, columns * thumbnail_width, 3), dtype=np.uint8)
  for index, image in enumerate(images):
    thumbnail = cv2.resize(image, (thumbnail_width, thumbnail_height), interpolation=cv2.INTER_AREA)
    row, column = divmod(index, columns)
    sheet[
        row * thumbnail_height:(row + 1) * thumbnail_height,
        column * thumbnail_width:(column + 1) * thumbnail_width,
    ] = thumbnail
  return sheet


def candidate_display_positions(stage: dict) -> np.ndarray:
  count = len(stage["candidate_ids"])
  if "scorer_rank" in stage:
    return np.asarray(stage["scorer_rank"]).reshape(-1).argsort()
  if "geometry_total_score" in stage:
    return np.asarray(stage["geometry_total_score"]).reshape(-1).argsort()
  if "axis_alignment" in stage:
    return np.asarray(stage["axis_alignment"]).reshape(-1).argsort()[::-1]
  return np.arange(count, dtype=np.int64)


def candidate_metric_value(stage: dict, key: str, position: int):
  if key not in stage:
    return None
  values = np.asarray(stage[key]).reshape(-1)
  if position >= len(values):
    return None
  value = values[position]
  if isinstance(value, np.generic):
    value = value.item()
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def add_candidate_label(image: np.ndarray, lines: list[str], selected: bool) -> np.ndarray:
  output = image.copy()
  line_height = 18
  panel_height = 8 + line_height * len(lines)
  overlay = output.copy()
  cv2.rectangle(overlay, (0, 0), (output.shape[1], panel_height), (0, 0, 0), thickness=-1)
  output = cv2.addWeighted(overlay, 0.65, output, 0.35, 0)
  color = (80, 255, 80) if selected else (255, 80, 80)
  for line_index, line in enumerate(lines):
    cv2.putText(
        output,
        line,
        (8, 18 + line_index * line_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        color,
        1,
        cv2.LINE_AA,
    )
  return output


def save_candidate_pipeline_diagnostics(
    result_dir: str,
    repeat_stem: str,
    tracker,
    color: np.ndarray,
    K: np.ndarray,
    target_mask: np.ndarray,
    diagnostics: dict,
    jpeg_quality: int,
) -> dict:
  output_dir = os.path.join(result_dir, "candidate_pipeline", repeat_stem)
  os.makedirs(output_dir, exist_ok=True)
  stages = list(diagnostics.get("stages", ()))
  stage_by_name = {str(stage["name"]): stage for stage in stages}
  target = np.asarray(target_mask, dtype=np.uint8) > 0
  tf_to_centered_mesh = np.asarray(
      tracker.estimator.get_tf_to_centered_mesh().detach().cpu().numpy(),
      dtype=np.float32,
  ).reshape(4, 4)
  stage_summaries = []
  npz_arrays = {}

  metric_keys = (
      "axis_alignment",
      "axis_angle_deg",
      "axis_rank",
      "axis_angle_pass",
      "geometry_center_error",
      "geometry_depth_error",
      "geometry_area_error",
      "geometry_aspect_error",
      "geometry_bbox_iou",
      "geometry_total_score",
      "geometry_rank",
      "scorer_score",
      "scorer_rank",
  )

  for stage_index, stage in enumerate(stages, start=1):
    stage_name = str(stage["name"])
    poses = np.asarray(stage["poses"], dtype=np.float32).reshape(-1, 4, 4)
    candidate_ids = np.asarray(stage["candidate_ids"], dtype=np.int64).reshape(-1)
    selected_mask = np.asarray(stage.get("selected_for_next", np.ones(len(poses), dtype=bool)), dtype=bool)
    display_positions = candidate_display_positions(stage)
    parent_stage = stage_by_name.get(stage.get("parent_stage"))
    parent_pose_by_id = {}
    if parent_stage is not None:
      parent_ids = np.asarray(parent_stage["candidate_ids"], dtype=np.int64).reshape(-1)
      parent_poses = np.asarray(parent_stage["poses"], dtype=np.float32).reshape(-1, 4, 4)
      parent_pose_by_id = {
          int(candidate_id): parent_poses[position]
          for position, candidate_id in enumerate(parent_ids)
      }

    records = [None] * len(poses)
    thumbnails = []
    for chunk_start in range(0, len(display_positions), 16):
      chunk_positions = display_positions[chunk_start:chunk_start + 16]
      chunk_masks = tracker._render_pose_masks(K, color.shape[:2], poses[chunk_positions])
      for chunk_index, position_value in enumerate(chunk_positions):
        position = int(position_value)
        candidate_id = int(candidate_ids[position])
        pose = poses[position]
        output_pose = pose @ tf_to_centered_mesh
        rendered_mask = np.asarray(chunk_masks[chunk_index]) > 0
        intersection = int(np.logical_and(rendered_mask, target).sum())
        union = int(np.logical_or(rendered_mask, target).sum())
        rendered_iou = float(intersection / union) if union > 0 else 0.0
        parent_pose = parent_pose_by_id.get(candidate_id)
        translation_update = None
        rotation_update = None
        if parent_pose is not None:
          translation_update = float(np.linalg.norm(pose[:3, 3] - parent_pose[:3, 3]))
          rotation_update = candidate_rotation_update_deg(parent_pose, pose)

        metrics = {
            key: candidate_metric_value(stage, key, position)
            for key in metric_keys
            if key in stage
        }
        record = {
            "candidate_id": candidate_id,
            "stage_position": position,
            "selected_for_next": bool(selected_mask[position]),
            "rendered_mask_iou": rendered_iou,
            "translation_update_m": translation_update,
            "rotation_update_deg": rotation_update,
            "centered_pose": pose.tolist(),
            "output_pose": output_pose.tolist(),
            **metrics,
        }
        records[position] = record

        visualization = tracker.draw_visualization(
            color,
            K,
            output_pose,
            rendered_mask=rendered_mask,
        )
        visualization[target] = (
            0.65 * visualization[target] + 0.35 * np.array([255, 0, 0])
        ).astype(np.uint8)
        lines = [
            f"{stage_name} ID:{candidate_id} {'KEEP' if selected_mask[position] else 'DROP'}",
            f"pos:{position} IoU:{rendered_iou:.3f}",
        ]
        axis_angle = metrics.get("axis_angle_deg")
        if axis_angle is not None:
          lines.append(
              f"PCA rank:{int(metrics['axis_rank'])} angle:{float(axis_angle):.2f} deg "
              f"align:{float(metrics['axis_alignment']):.4f}"
          )
        geometry_score = metrics.get("geometry_total_score")
        if geometry_score is not None:
          lines.append(f"Geometry rank:{int(metrics['geometry_rank'])} score:{float(geometry_score):.4f}")
        scorer_score = metrics.get("scorer_score")
        if scorer_score is not None:
          lines.append(f"Scorer rank:{int(metrics['scorer_rank'])} score:{float(scorer_score):.5f}")
        if translation_update is not None:
          lines.append(f"update:{translation_update * 1000.0:.2f} mm / {rotation_update:.2f} deg")
        thumbnails.append(add_candidate_label(visualization, lines, bool(selected_mask[position])))

    filename_stem = f"{stage_index:02d}_{stage_name}"
    contact_path = os.path.join(output_dir, f"{filename_stem}_contact_sheet.jpg")
    contact = candidate_contact_sheet(thumbnails)
    if not cv2.imwrite(
        contact_path,
        contact[..., ::-1],
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    ):
      raise RuntimeError(f"failed to save candidate contact sheet: {contact_path}")

    stage_metadata = {
        key: value
        for key, value in stage.items()
        if key not in {"poses", "candidate_ids", "selected_candidate_ids", "selected_for_next", *metric_keys}
    }
    stage_summaries.append({
        "name": stage_name,
        "parent_stage": stage.get("parent_stage"),
        "candidate_count": len(poses),
        "selected_count": int(selected_mask.sum()),
        "contact_sheet": contact_path,
        "metadata": serializable_foundation_timing(stage_metadata),
        "candidates": records,
    })
    npz_prefix = f"stage_{stage_index:02d}_{stage_name}"
    npz_arrays[f"{npz_prefix}__poses"] = poses
    npz_arrays[f"{npz_prefix}__output_poses"] = np.asarray(
      [record["output_pose"] for record in records],
      dtype=np.float32,
    )
    npz_arrays[f"{npz_prefix}__candidate_ids"] = candidate_ids
    npz_arrays[f"{npz_prefix}__selected_for_next"] = selected_mask
    npz_arrays[f"{npz_prefix}__rendered_mask_iou"] = np.asarray(
      [record["rendered_mask_iou"] for record in records],
      dtype=np.float32,
    )
    npz_arrays[f"{npz_prefix}__translation_update_m"] = np.asarray(
      [np.nan if record["translation_update_m"] is None else record["translation_update_m"] for record in records],
      dtype=np.float32,
    )
    npz_arrays[f"{npz_prefix}__rotation_update_deg"] = np.asarray(
      [np.nan if record["rotation_update_deg"] is None else record["rotation_update_deg"] for record in records],
      dtype=np.float32,
    )
    for key in metric_keys:
      if key in stage:
        npz_arrays[f"{npz_prefix}__{key}"] = np.asarray(stage[key])

  json_path = os.path.join(output_dir, "candidate_pipeline.json")
  npz_path = os.path.join(output_dir, "candidate_pipeline.npz")
  with open(json_path, "w", encoding="utf-8") as file:
    json.dump({
        "version": diagnostics.get("version", 1),
        "init_strategy": diagnostics.get("init_strategy"),
        "top1_candidate_id": diagnostics.get("top1_candidate_id"),
        "coarse_scorer_status": diagnostics.get("coarse_scorer_status"),
        "axis_prior": serializable_foundation_timing(diagnostics.get("axis_prior", {})),
        "final_ranked_candidate_ids": serializable_foundation_timing(diagnostics.get("final_ranked_candidate_ids", [])),
        "final_ranked_scores": serializable_foundation_timing(diagnostics.get("final_ranked_scores", [])),
        "stages": stage_summaries,
    }, file, indent=2, ensure_ascii=False, allow_nan=False)
  np.savez_compressed(npz_path, **npz_arrays)
  return {
      "directory": output_dir,
      "json": json_path,
      "npz": npz_path,
      "contact_sheets": [stage["contact_sheet"] for stage in stage_summaries],
  }


def timing_statistics(values: list[float]) -> dict:
  array = np.asarray(values, dtype=np.float64)
  return {
      "mean_seconds": float(array.mean()),
      "min_seconds": float(array.min()),
      "max_seconds": float(array.max()),
      "std_seconds": float(array.std()),
  }


def main() -> None:
  args = parse_args()
  input_path = os.path.abspath(args.input)
  config_path, output_dir = infer_paths(input_path, args.config, args.output_dir)
  os.makedirs(output_dir, exist_ok=True)

  publisher = RecordedFramePublisher(input_path)
  cfg = load_config(config_path)
  foundation_cfg = cfg.get("foundationpose", {})
  simulation_cfg = cfg.get("simulation", {})
  candidate_pipeline_debug_enabled = bool(simulation_cfg.get("candidate_pipeline_debug_enabled", False))
  tracker = build_tracker(
      foundation_cfg,
      candidate_pipeline_debug_enabled=candidate_pipeline_debug_enabled,
  )
  runtime_cfg = cfg.get("runtime", {})
  quality_thresholds = {
      "register_min_render_iou": float(runtime_cfg.get("register_min_render_iou", 0.0)),
      "register_uncertain_score_gap": float(runtime_cfg.get("register_uncertain_score_gap", 0.0)),
      "register_uncertain_render_iou": float(runtime_cfg.get("register_uncertain_render_iou", 0.0)),
      "register_max_translation_drift": float(runtime_cfg.get("register_max_translation_drift", 0.0)),
  }
  print(f"[SimCamera][CONFIG] path={config_path}")
  print(
      "[SimCamera][CANDIDATE_PIPELINE] "
      f"enabled={candidate_pipeline_debug_enabled}"
  )
  print(
      "[SimCamera][AXIS_VIS] "
      f"enabled={tracker.axis_prior_visualization_enabled} "
      f"axis_prior_enabled={tracker.axis_prior_filter == 'depth_pca'} "
      f"output={tracker.axis_prior_visualization_dir} "
      f"top_n={tracker.axis_prior_visualization_top_n} "
      f"boundary_margin={tracker.axis_prior_visualization_boundary_margin} "
      f"max_records={tracker.axis_prior_visualization_max_records}"
  )
  print(
      "[SimCamera][QUALITY] "
      f"min_render_iou={quality_thresholds['register_min_render_iou']} "
      f"uncertain_score_gap={quality_thresholds['register_uncertain_score_gap']} "
      f"uncertain_render_iou={quality_thresholds['register_uncertain_render_iou']} "
      f"max_translation_drift={quality_thresholds['register_max_translation_drift']}"
  )
  recording_cfg = cfg.get("recording", {})
  save_topk_poses = bool(recording_cfg.get("save_topk_poses", True))
  topk_pose_count = int(recording_cfg.get("topk_pose_count", 5))
  jpeg_quality = max(1, min(int(recording_cfg.get("jpeg_quality", 90)), 100))
  repeat = max(1, int(args.repeat))
  results = []
  stem = os.path.splitext(os.path.basename(input_path))[0]
  result_dir = os.path.join(output_dir, stem)
  if os.path.isdir(result_dir):
    shutil.rmtree(result_dir)
  os.makedirs(result_dir, exist_ok=True)
  detection = publisher.detection()
  stable_areas = []
  required_stable_detections = max(1, int(runtime_cfg.get("register_required_stable_detections", 1)))
  for _ in range(required_stable_detections - 1):
    register_stability_reject_reason(
        detection,
        stable_areas,
        required_stable_detections,
        float(runtime_cfg.get("register_max_area_change", 0.0)),
    )

  for index in range(repeat):
    repeat_start = time.perf_counter()
    repeat_index = index + 1
    simulated_frame_id = publisher.frame_id + index
    publisher.publish(simulated_frame_id)
    frame, mask = receive_synchronized_input(publisher)
    validation_start = time.perf_counter()
    validation_reject_reason, valid_depth_pixels = replay_validation_reject_reason(
        detection=detection,
        depth=frame.depth,
        image_shape=frame.color.shape[:2],
        runtime_cfg=runtime_cfg,
        stable_areas=stable_areas,
    )
    validation_elapsed = time.perf_counter() - validation_start
    if validation_reject_reason is not None:
      raise RuntimeError(f"replay input rejected before register: {validation_reject_reason}")
    tracker.reset()
    start = time.perf_counter()
    pose_result = tracker.register(
      frame.color,
      frame.depth,
      frame.K,
      mask.mask,
      frame_id=frame.frame_id,
      timestamp=frame.timestamp,
      sequence_id=os.path.basename(input_path),
      capture_source="recorded_replay",
    )
    elapsed = time.perf_counter() - start
    new_pose = np.asarray(pose_result.pose, dtype=np.float32).reshape(4, 4)
    translation_error = float(np.linalg.norm(new_pose[:3, 3] - publisher.saved_output_pose[:3, 3]))
    rotation_error = rotation_error_deg(publisher.saved_output_pose, new_pose)
    repeat_stem = f"repeat_{repeat_index:03d}"
    topk_contact_sheet = None
    topk_elapsed = 0.0
    if save_topk_poses:
      topk_start = time.perf_counter()
      topk_record = build_topk_pose_record(
        tracker=tracker,
        color=frame.color,
        K=frame.K,
        target_mask=mask.mask,
        frame_id=frame.frame_id,
        count=topk_pose_count,
      )
      if topk_record is not None:
        topk_contact_sheet = save_topk_repeat(
            result_dir=result_dir,
            repeat_stem=repeat_stem,
            topk_record=topk_record,
            jpeg_quality=jpeg_quality,
            repeat_index=repeat_index,
            source_frame_id=publisher.frame_id,
            simulated_frame_id=simulated_frame_id,
        )
      topk_elapsed = time.perf_counter() - topk_start
    candidate_pipeline_record = None
    candidate_pipeline_elapsed = 0.0
    if candidate_pipeline_debug_enabled:
      candidate_pipeline_start = time.perf_counter()
      candidate_diagnostics = getattr(tracker.estimator, "last_candidate_diagnostics", None)
      if not candidate_diagnostics:
        raise RuntimeError("candidate pipeline diagnostics were enabled but no registration snapshot was produced")
      candidate_pipeline_record = save_candidate_pipeline_diagnostics(
          result_dir=result_dir,
          repeat_stem=repeat_stem,
          tracker=tracker,
          color=frame.color,
          K=frame.K,
          target_mask=mask.mask,
          diagnostics=candidate_diagnostics,
          jpeg_quality=jpeg_quality,
      )
      candidate_pipeline_elapsed = time.perf_counter() - candidate_pipeline_start
    quality_start = time.perf_counter()
    quality_result = evaluate_register_quality(
      tracker=tracker,
      pose=new_pose,
      depth=frame.depth,
      K=frame.K,
      mask=mask.mask,
      runtime_cfg=runtime_cfg,
    )
    quality_elapsed = time.perf_counter() - quality_start
    repeat_elapsed = time.perf_counter() - repeat_start
    timing = dict(getattr(tracker.estimator, "last_register_timing", {}))
    print_foundation_candidate_summary(repeat_index, timing)
    print_timing_summary(
        repeat_index,
        [
            ("replay_validation", seconds_to_ms(validation_elapsed)),
            ("", None),
            *foundation_register_timing_rows(timing, elapsed),
            ("", None),
            ("register_quality_gate", seconds_to_ms(quality_elapsed)),
            ("top5_diagnostics", seconds_to_ms(topk_elapsed)),
            ("candidate_pipeline_diagnostics", seconds_to_ms(candidate_pipeline_elapsed)),
            ("repeat_total", seconds_to_ms(repeat_elapsed)),
        ],
    )
    visualization_start = time.perf_counter()
    visualization = tracker.draw_visualization(
        frame.color,
        frame.K,
        new_pose,
        rendered_mask=(
            quality_result.rendered_mask
            if tracker.quality_render_mask_reuse_enabled else None
        ),
    )
    visualization_elapsed = time.perf_counter() - visualization_start
    mask_pixels = mask.mask.astype(bool)
    visualization[mask_pixels] = (
        0.65 * visualization[mask_pixels] + 0.35 * np.array([255, 0, 0])
    ).astype(np.uint8)
    confidence = mask.confidence if np.isfinite(mask.confidence) else None
    visualization = add_pose_record_label(
        visualization,
        frame.frame_id,
      (
        f"register {repeat_index}/{repeat} ACCEPTED"
        if quality_result.accepted
        else f"register {repeat_index}/{repeat} REJECTED:{quality_result.reject_reason}"
      ),
        confidence,
    )
    image_path = os.path.join(result_dir, f"{repeat_stem}.jpg")
    pose_path = os.path.join(result_dir, f"{repeat_stem}_pose.txt")
    if not cv2.imwrite(image_path, visualization[..., ::-1]):
      raise RuntimeError(f"failed to save simulated result: {image_path}")
    np.savetxt(pose_path, new_pose)
    results.append({
        "repeat_index": repeat_index,
        "source_frame_id": publisher.frame_id,
        "simulated_frame_id": simulated_frame_id,
        "accepted": quality_result.accepted,
        "reject_reason": quality_result.reject_reason,
        "valid_depth_pixels": valid_depth_pixels,
        "render_iou": quality_result.render_iou,
        "translation_drift": quality_result.translation_drift,
        "top1_top2_score_gap": quality_result.top1_top2_score_gap,
        "validation_seconds": validation_elapsed,
        "elapsed_seconds": elapsed,
        "quality_gate_seconds": quality_elapsed,
        "visualization_seconds": visualization_elapsed,
        "top5_diagnostics_seconds": topk_elapsed,
        "candidate_pipeline_diagnostics_seconds": candidate_pipeline_elapsed,
        "repeat_total_seconds": repeat_elapsed,
        "foundation_timing_seconds": serializable_foundation_timing(timing),
        "translation_difference_from_saved_m": translation_error,
        "rotation_difference_from_saved_deg": rotation_error,
        "pose": new_pose.tolist(),
        "visualization": image_path,
        "pose_file": pose_path,
        "topk_contact_sheet": topk_contact_sheet,
        "candidate_pipeline": candidate_pipeline_record,
    })
    print(
        f"[SimCamera] {repeat_index}/{repeat} source_frame={publisher.frame_id} simulated_frame={frame.frame_id} "
        f"accepted={quality_result.accepted} reject_reason={quality_result.reject_reason} "
        f"render_iou={quality_result.render_iou} score_gap={quality_result.top1_top2_score_gap} "
        f"translation_difference={translation_error:.6f}m "
        f"rotation_difference={rotation_error:.4f}deg time={elapsed:.3f}s"
    )

  timing_summary = {
      "validation": timing_statistics([item["validation_seconds"] for item in results]),
      "foundation_register": timing_statistics([item["elapsed_seconds"] for item in results]),
      "quality_gate": timing_statistics([item["quality_gate_seconds"] for item in results]),
      "visualization": timing_statistics([item["visualization_seconds"] for item in results]),
      "top5_diagnostics": timing_statistics([item["top5_diagnostics_seconds"] for item in results]),
      "candidate_pipeline_diagnostics": timing_statistics([item["candidate_pipeline_diagnostics_seconds"] for item in results]),
      "repeat_total": timing_statistics([item["repeat_total_seconds"] for item in results]),
  }
  accepted_count = sum(1 for item in results if item["accepted"])
  print("[SimCamera][SUMMARY]")
  print(f"  accepted={accepted_count}/{repeat} rejected={repeat - accepted_count}/{repeat}")
  for stage, statistics in timing_summary.items():
    print(
        f"  {stage}: mean={statistics['mean_seconds'] * 1000.0:.3f}ms "
        f"min={statistics['min_seconds'] * 1000.0:.3f}ms "
        f"max={statistics['max_seconds'] * 1000.0:.3f}ms "
        f"std={statistics['std_seconds'] * 1000.0:.3f}ms"
    )

  comparison_path = os.path.join(result_dir, "comparison.json")
  with open(comparison_path, "w", encoding="utf-8") as file:
    json.dump({
        "input": input_path,
        "config": config_path,
        "quality_thresholds": quality_thresholds,
        "source_frame_id": publisher.frame_id,
        "repeat": repeat,
        "accepted_count": accepted_count,
        "rejected_count": repeat - accepted_count,
        "timing_summary": timing_summary,
        "saved_output_pose": publisher.saved_output_pose.tolist(),
        "results": results,
    }, file, indent=2, ensure_ascii=False)

  print(f"[SimCamera] repeated register results: {result_dir}")
  print(f"[SimCamera] comparison: {comparison_path}")


if __name__ == "__main__":
  main()

# python realtime_foundation/simulate_recorded_camera.py \
#  --input realtime_foundation/outputs/frame_records/frame_data/frame_000057_register.npz

# 重复运行同一帧，测试同一输入是否偶发选择不同姿态：
# python simulate_recorded_camera.py --input ./outputs/error/frame_001095_register.npz --config ./config.yaml --repeat 20