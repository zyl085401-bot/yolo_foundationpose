from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import trimesh


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FOUNDATIONPOSE_DIR = os.path.join(REPO_ROOT, "FoundationPose")
if FOUNDATIONPOSE_DIR not in sys.path:
  sys.path.insert(0, FOUNDATIONPOSE_DIR)

from estimater import (  # noqa: E402
    FoundationPose,
    PoseRefinePredictor,
    ScorePredictor,
    dr,
    draw_posed_3d_box,
    draw_xyz_axis,
    nvdiffrast_render,
    nvdiffrast_render_mask,
    set_logging_format,
    set_seed,
)
import learning.training.predict_pose_refine as pose_refine_module  # noqa: E402
import learning.training.predict_score as score_module  # noqa: E402


_ORIGINAL_COMPUTE_CROP_WINDOW_TF_BATCH = pose_refine_module.compute_crop_window_tf_batch


def _compute_crop_window_tf_batch_float32(*, pts, H, W, poses, K, crop_ratio, out_size, method, mesh_diameter=None):
  if torch.is_tensor(pts):
    pts = pts.to(device="cuda", dtype=torch.float32)
  else:
    pts = torch.as_tensor(np.asarray(pts, dtype=np.float32), device="cuda", dtype=torch.float32)

  if torch.is_tensor(poses):
    poses = poses.to(device="cuda", dtype=torch.float32)
  else:
    poses = torch.as_tensor(np.asarray(poses, dtype=np.float32), device="cuda", dtype=torch.float32)

  if torch.is_tensor(K):
    K = K.to(device="cuda", dtype=torch.float32)
  else:
    K = torch.as_tensor(np.asarray(K, dtype=np.float32), device="cuda", dtype=torch.float32)

  if method != "box_3d":
    return _ORIGINAL_COMPUTE_CROP_WINDOW_TF_BATCH(
        pts=pts,
        H=H,
        W=W,
        poses=poses,
        K=K,
        crop_ratio=float(crop_ratio),
        out_size=out_size,
        method=method,
        mesh_diameter=float(mesh_diameter) if mesh_diameter is not None else None,
    )

  batch_size = len(poses)
  radius = float(mesh_diameter) * float(crop_ratio) / 2.0
  offsets = torch.tensor(
      [
          [0.0, 0.0, 0.0],
          [radius, 0.0, 0.0],
          [-radius, 0.0, 0.0],
          [0.0, radius, 0.0],
          [0.0, -radius, 0.0],
      ],
      device="cuda",
      dtype=torch.float32,
  )
  crop_pts = poses[:, :3, 3].reshape(-1, 1, 3) + offsets.reshape(1, -1, 3)
  projected = (K.reshape(3, 3) @ crop_pts.reshape(-1, 3).T).T
  uvs = projected[:, :2] / projected[:, 2:3]
  uvs = uvs.reshape(batch_size, -1, 2)
  center = uvs[:, 0]
  radius_px = torch.abs(uvs - center.reshape(-1, 1, 2)).reshape(batch_size, -1).max(axis=-1)[0].reshape(-1)
  left = center[:, 0] - radius_px
  right = center[:, 0] + radius_px
  top = center[:, 1] - radius_px
  bottom = center[:, 1] + radius_px
  return _compute_crop_tf_batch_float32(left, right, top, bottom, out_size)


def _compute_crop_tf_batch_float32(left: torch.Tensor, right: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor, out_size) -> torch.Tensor:
  batch_size = len(left)
  left = left.round()
  right = right.round()
  top = top.round()
  bottom = bottom.round()

  tf = torch.eye(3, device="cuda", dtype=torch.float32)[None].expand(batch_size, -1, -1).contiguous()
  tf[:, 0, 2] = -left
  tf[:, 1, 2] = -top
  new_tf = torch.eye(3, device="cuda", dtype=torch.float32)[None].expand(batch_size, -1, -1).contiguous()
  new_tf[:, 0, 0] = float(out_size[0]) / (right - left)
  new_tf[:, 1, 1] = float(out_size[1]) / (bottom - top)
  return new_tf @ tf


def _set_crop_window_patch(enabled: bool) -> None:
  crop_fn = _compute_crop_window_tf_batch_float32 if enabled else _ORIGINAL_COMPUTE_CROP_WINDOW_TF_BATCH
  pose_refine_module.compute_crop_window_tf_batch = crop_fn
  score_module.compute_crop_window_tf_batch = crop_fn


@dataclass
class PoseResult:
  pose: np.ndarray
  initialized: bool
  mode: str


class FoundationPoseRealtimeTracker:
  def __init__(
      self,
      mesh_file: str,
      debug_dir: str,
      debug: int = 1,
      use_float32_crop_window_patch: bool = True,
      init_min_n_views: int = 40,
      init_inplane_step: int = 60,
      est_refine_iter: int = 5,
      init_strategy: str = "default",
      coarse_refine_iter: int = 1,
      coarse_score_filter: str = "none",
      coarse_score_top_k: int = 999999,
      fine_refine_iter: int = 2,
      fine_top_k: int = 16,
      axis_prior_filter: str = "none",
      axis_prior_model_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
      axis_prior_max_angle_deg: float = 45.0,
      axis_prior_min_candidates: int = 12,
      axis_prior_max_candidates: int = 0,
      axis_prior_min_points: int = 500,
      axis_prior_min_confidence: float = 1.4,
      axis_prior_visualization_enabled: bool = False,
      axis_prior_visualization_dir: str | None = None,
      axis_prior_visualization_top_n: int = 24,
      axis_prior_visualization_boundary_margin: int = 6,
      axis_prior_visualization_max_records: int = 20,
      network_input_capture: dict | None = None,
      render_profile_enabled: bool = False,
      render_batched_matmul_enabled: bool = False,
      scorer_precomputed_xyz_enabled: bool = False,
      refiner_stage1_optimizations_enabled: bool = False,
      refiner_shared_warp_grid_enabled: bool = False,
      scorer_shared_warp_grid_enabled: bool = False,
      scorer_skip_unused_depth_warp_enabled: bool = False,
      tensorrt_backends: dict | None = None,
      track_refine_iter: int = 2,
      vis_mode: str = "box",
      contour_thickness: int = 3,
      axis_scale: float = 0.1,
      skip_redundant_coarse_scorer: bool = False,
      network_internal_sync_enabled: bool = True,
      frame_statistics_reuse_enabled: bool = False,
      quality_render_mask_reuse_enabled: bool = False,
  ):
    set_logging_format()
    set_seed(0)

    self.mesh_file = mesh_file
    self.debug_dir = debug_dir
    self.debug = debug
    self.use_float32_crop_window_patch = use_float32_crop_window_patch
    self.init_min_n_views = init_min_n_views
    self.init_inplane_step = init_inplane_step
    self.est_refine_iter = est_refine_iter
    self.init_strategy = init_strategy
    self.coarse_refine_iter = coarse_refine_iter
    self.coarse_score_filter = coarse_score_filter
    self.coarse_score_top_k = coarse_score_top_k
    self.fine_refine_iter = fine_refine_iter
    self.fine_top_k = fine_top_k
    self.skip_redundant_coarse_scorer = skip_redundant_coarse_scorer
    self.frame_statistics_reuse_enabled = frame_statistics_reuse_enabled
    self.quality_render_mask_reuse_enabled = quality_render_mask_reuse_enabled
    self.axis_prior_filter = axis_prior_filter
    self.axis_prior_model_axis = axis_prior_model_axis
    self.axis_prior_max_angle_deg = axis_prior_max_angle_deg
    self.axis_prior_min_candidates = axis_prior_min_candidates
    self.axis_prior_max_candidates = axis_prior_max_candidates
    self.axis_prior_min_points = axis_prior_min_points
    self.axis_prior_min_confidence = axis_prior_min_confidence
    self.axis_prior_visualization_enabled = axis_prior_visualization_enabled
    self.axis_prior_visualization_dir = axis_prior_visualization_dir or os.path.join(debug_dir, "axis_prior_debug")
    self.axis_prior_visualization_top_n = max(1, axis_prior_visualization_top_n)
    self.axis_prior_visualization_boundary_margin = max(0, axis_prior_visualization_boundary_margin)
    self.axis_prior_visualization_max_records = max(0, axis_prior_visualization_max_records)
    self._axis_prior_visualization_run_id = time.strftime("run_%Y%m%d_%H%M%S")
    self._axis_prior_visualization_count = 0
    self.track_refine_iter = track_refine_iter
    self.vis_mode = vis_mode
    self.contour_thickness = contour_thickness
    self.axis_scale = axis_scale
    self.initialized = False
    self.last_pose = None

    _set_crop_window_patch(self.use_float32_crop_window_patch)

    os.makedirs(self.debug_dir, exist_ok=True)
    self.mesh = trimesh.load(mesh_file)
    self.mesh.vertices = np.asarray(self.mesh.vertices, dtype=np.float32)
    self.model_normals = np.asarray(self.mesh.vertex_normals, dtype=np.float32)
    self.to_origin, self.extents = trimesh.bounds.oriented_bounds(self.mesh)
    self.to_origin = np.asarray(self.to_origin, dtype=np.float32)
    self.extents = np.asarray(self.extents, dtype=np.float32)
    self.bbox = np.stack([-self.extents / 2, self.extents / 2], axis=0).reshape(2, 3).astype(np.float32)

    tensorrt_backends = dict(tensorrt_backends or {})
    self.scorer = ScorePredictor(
      network_input_capture=network_input_capture,
      tensorrt_backend=tensorrt_backends.get("scorer"),
      render_profile_enabled=render_profile_enabled,
      render_batched_matmul_enabled=render_batched_matmul_enabled,
      scorer_precomputed_xyz_enabled=scorer_precomputed_xyz_enabled,
      scorer_shared_warp_grid_enabled=scorer_shared_warp_grid_enabled,
      scorer_skip_unused_depth_warp_enabled=scorer_skip_unused_depth_warp_enabled,
      network_internal_sync_enabled=network_internal_sync_enabled,
    )
    self.refiner = PoseRefinePredictor(
      network_input_capture=network_input_capture,
      tensorrt_backend=tensorrt_backends.get("refiner"),
      render_profile_enabled=render_profile_enabled,
      render_batched_matmul_enabled=render_batched_matmul_enabled,
      refiner_stage1_optimizations_enabled=refiner_stage1_optimizations_enabled,
      refiner_shared_warp_grid_enabled=refiner_shared_warp_grid_enabled,
      network_internal_sync_enabled=network_internal_sync_enabled,
    )
    self.glctx = dr.RasterizeCudaContext()
    self.estimator = FoundationPose(
        model_pts=self.mesh.vertices,
        model_normals=self.model_normals,
        mesh=self.mesh,
        scorer=self.scorer,
        refiner=self.refiner,
        debug_dir=self.debug_dir,
        debug=self.debug,
        glctx=self.glctx,
        init_min_n_views=self.init_min_n_views,
        init_inplane_step=self.init_inplane_step,
    )
    self._ensure_estimator_float32()

  def reset(self) -> None:
    self.initialized = False
    self.last_pose = None
    self.estimator.pose_last = None

  def register(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray, mask: np.ndarray) -> PoseResult:
    color, depth, K = self._prepare_frame_inputs(color, depth, K)
    mask = self._valid_mask(mask, depth)
    pose = self.estimator.register(
        K=K,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=self.est_refine_iter,
        init_strategy=self.init_strategy,
        coarse_refine_iter=self.coarse_refine_iter,
        coarse_score_filter=self.coarse_score_filter,
        coarse_score_top_k=self.coarse_score_top_k,
        fine_refine_iter=self.fine_refine_iter,
        fine_top_k=self.fine_top_k,
        skip_redundant_coarse_scorer=self.skip_redundant_coarse_scorer,
        axis_prior_filter=self.axis_prior_filter,
        axis_prior_model_axis=self.axis_prior_model_axis,
        axis_prior_max_angle_deg=self.axis_prior_max_angle_deg,
        axis_prior_min_candidates=self.axis_prior_min_candidates,
        axis_prior_max_candidates=self.axis_prior_max_candidates,
        axis_prior_min_points=self.axis_prior_min_points,
        axis_prior_min_confidence=self.axis_prior_min_confidence,
        axis_prior_debug=self.axis_prior_visualization_enabled,
        frame_statistics_reuse_enabled=self.frame_statistics_reuse_enabled,
    )
    if self.axis_prior_visualization_enabled:
      try:
        self._save_axis_prior_visualization(color, K, mask)
      except Exception:
        logging.exception("Failed to save axis-prior candidate visualization")
    self.initialized = True
    self.last_pose = pose
    return PoseResult(pose=pose, initialized=True, mode="register")

  def track(self, color: np.ndarray, depth: np.ndarray, K: np.ndarray) -> PoseResult:
    if not self.initialized:
      raise RuntimeError("FoundationPoseRealtimeTracker.track called before register")
    color, depth, K = self._prepare_frame_inputs(color, depth, K)
    pose = self.estimator.track_one(rgb=color, depth=depth, K=K, iteration=self.track_refine_iter)
    self.last_pose = pose
    return PoseResult(pose=pose, initialized=True, mode="track")

  def render_pose_mask(self, K: np.ndarray, image_shape: tuple[int, int], pose: np.ndarray | None = None) -> np.ndarray:
    pose_to_render = self.estimator.pose_last if pose is None else pose
    K = np.ascontiguousarray(K, dtype=np.float32)
    height, width = image_shape[:2]
    ob_in_cams = torch.as_tensor(pose_to_render, device="cuda", dtype=torch.float).reshape(1, 4, 4)
    rendered_masks = nvdiffrast_render_mask(
        K=K,
        H=height,
        W=width,
        ob_in_cams=ob_in_cams,
        glctx=self.glctx,
        mesh_tensors=self.estimator.mesh_tensors,
    )
    return rendered_masks[0]

  def _render_pose_masks(self, K: np.ndarray, image_shape: tuple[int, int], poses: np.ndarray) -> np.ndarray:
    K = np.ascontiguousarray(K, dtype=np.float32)
    poses = np.ascontiguousarray(poses, dtype=np.float32).reshape(-1, 4, 4)
    height, width = image_shape[:2]
    ob_in_cams = torch.as_tensor(poses, device="cuda", dtype=torch.float32)
    _, render_depth, _ = nvdiffrast_render(
        K=K,
        H=height,
        W=width,
        ob_in_cams=ob_in_cams,
        glctx=self.glctx,
        mesh_tensors=self.estimator.mesh_tensors,
        output_size=np.asarray([height, width]),
        use_light=False,
    )
    return (render_depth.detach().cpu().numpy() > 0.001).astype(np.uint8)

  def _save_axis_prior_visualization(self, color: np.ndarray, K: np.ndarray, target_mask: np.ndarray) -> None:
    if self._axis_prior_visualization_count >= self.axis_prior_visualization_max_records:
      return
    diagnostics = getattr(self.estimator, "last_axis_prior_diagnostics", None)
    if not diagnostics:
      logging.warning("Axis-prior visualization skipped because no valid PCA diagnostics were produced")
      return

    poses = np.asarray(diagnostics["poses_before"], dtype=np.float32)
    ranked = np.asarray(diagnostics["ranked_indices"], dtype=np.int64)
    kept_indices = np.asarray(diagnostics["kept_indices"], dtype=np.int64)
    kept_set = set(int(index) for index in kept_indices)
    kept_ranked = np.asarray([index for index in ranked if int(index) in kept_set], dtype=np.int64)
    top_indices = ranked[:min(self.axis_prior_visualization_top_n, len(ranked))]
    kept_count = len(kept_indices)
    margin = self.axis_prior_visualization_boundary_margin
    boundary_start = max(0, kept_count - margin)
    boundary_end = min(len(ranked), kept_count + margin)
    boundary_indices = ranked[boundary_start:boundary_end]

    selected_indices = list(dict.fromkeys(
        [int(index) for index in np.concatenate((top_indices, boundary_indices, kept_ranked))]
    ))
    if not selected_indices:
      return
    masks = self._render_pose_masks(K, color.shape[:2], poses[selected_indices])
    mask_by_index = {index: masks[position] for position, index in enumerate(selected_indices)}
    rank_by_index = {int(index): rank for rank, index in enumerate(ranked, start=1)}
    target = np.asarray(target_mask, dtype=np.uint8) > 0
    iou_by_index = {}
    for index in selected_indices:
      rendered = mask_by_index[index] > 0
      intersection = int(np.logical_and(rendered, target).sum())
      union = int(np.logical_or(rendered, target).sum())
      iou_by_index[index] = float(intersection / union) if union > 0 else 0.0
    tf_to_centered_mesh = np.asarray(
        self.estimator.get_tf_to_centered_mesh().detach().cpu().numpy(),
        dtype=np.float32,
    ).reshape(4, 4)

    def make_tile(index: int) -> np.ndarray:
      centered_pose = poses[index]
      output_pose = centered_pose @ tf_to_centered_mesh
      center_pose = output_pose @ np.linalg.inv(self.to_origin)
      vis = np.ascontiguousarray(color, dtype=np.uint8).copy()
      vis[target] = (0.65 * vis[target] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
      contour_mask = mask_by_index[index] * 255
      contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      is_kept = index in kept_set
      status_color = (0, 255, 0) if is_kept else (255, 0, 0)
      if contours:
        cv2.drawContours(vis, contours, -1, color=status_color, thickness=self.contour_thickness, lineType=cv2.LINE_AA)
      vis = draw_xyz_axis(
          vis,
          ob_in_cam=center_pose,
          scale=self.axis_scale,
          K=K,
          thickness=3,
          transparency=0,
          is_input_rgb=True,
      )
      angle = float(diagnostics["angles_deg"][index])
      alignment = float(diagnostics["alignment"][index])
      angle_passed = bool(diagnostics["angle_pass_mask"][index])
      lines = [
          f"rank={rank_by_index[index]} angle={angle:.2f} deg",
          f"align={alignment:.4f} mask_iou={iou_by_index[index]:.3f}",
          f"{'KEEP' if is_kept else 'DROP'} angle_pass={'Y' if angle_passed else 'N'}",
      ]
      for line_number, text in enumerate(lines):
        y = 28 + line_number * 26
        cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 1, cv2.LINE_AA)
      cv2.rectangle(vis, (1, 1), (vis.shape[1] - 2, vis.shape[0] - 2), status_color, 4)
      return vis

    tile_by_index = {index: make_tile(index) for index in selected_indices}

    def make_image_contact_sheet(images, columns: int = 4, thumbnail_width: int = 320) -> np.ndarray:
      if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)
      source_height, source_width = images[0].shape[:2]
      thumbnail_height = max(1, int(round(source_height * thumbnail_width / source_width)))
      rows = (len(images) + columns - 1) // columns
      sheet = np.zeros((rows * thumbnail_height, columns * thumbnail_width, 3), dtype=np.uint8)
      for position, image in enumerate(images):
        thumbnail = cv2.resize(image, (thumbnail_width, thumbnail_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(position, columns)
        sheet[row * thumbnail_height:(row + 1) * thumbnail_height,
              column * thumbnail_width:(column + 1) * thumbnail_width] = thumbnail
      return sheet

    def make_contact_sheet(indices) -> np.ndarray:
      return make_image_contact_sheet([tile_by_index[int(index)] for index in indices])

    refined_tiles = []
    refined_poses = diagnostics.get("poses_after_coarse_refiner")
    if refined_poses is not None:
      refined_poses = np.asarray(refined_poses, dtype=np.float32).reshape(-1, 4, 4)
      refined_masks = self._render_pose_masks(K, color.shape[:2], refined_poses)
      selected_positions = set(int(value) for value in diagnostics.get("coarse_selected_positions", []))
      coarse_scores = np.asarray(diagnostics.get("coarse_scores", []), dtype=np.float32).reshape(-1)
      score_by_position = {
          position: float(coarse_scores[score_index])
          for score_index, position in enumerate(diagnostics.get("coarse_selected_positions", []))
          if score_index < len(coarse_scores)
      }
      for position, (centered_pose, rendered_mask) in enumerate(zip(refined_poses, refined_masks)):
        original_index = int(kept_indices[position])
        output_pose = centered_pose @ tf_to_centered_mesh
        center_pose = output_pose @ np.linalg.inv(self.to_origin)
        vis = np.ascontiguousarray(color, dtype=np.uint8).copy()
        vis[target] = (0.65 * vis[target] + 0.35 * np.array([255, 0, 0])).astype(np.uint8)
        rendered = rendered_mask > 0
        intersection = int(np.logical_and(rendered, target).sum())
        union = int(np.logical_or(rendered, target).sum())
        rendered_iou = float(intersection / union) if union > 0 else 0.0
        selected = position in selected_positions
        status_color = (0, 255, 0) if selected else (255, 255, 0)
        contours, _ = cv2.findContours(rendered_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
          cv2.drawContours(vis, contours, -1, color=status_color, thickness=self.contour_thickness, lineType=cv2.LINE_AA)
        vis = draw_xyz_axis(
            vis,
            ob_in_cam=center_pose,
            scale=self.axis_scale,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        score = score_by_position.get(position)
        lines = [
            f"axis_rank={rank_by_index[original_index]} after coarse Refiner",
            f"mask_iou={rendered_iou:.3f} {'TO_SCORER' if selected else 'GEOMETRY_DROP'}",
            f"coarse_score={'N/A' if score is None else f'{score:.4f}'}",
        ]
        for line_number, text in enumerate(lines):
          y = 28 + line_number * 26
          cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 4, cv2.LINE_AA)
          cv2.putText(vis, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 1, cv2.LINE_AA)
        cv2.rectangle(vis, (1, 1), (vis.shape[1] - 2, vis.shape[0] - 2), status_color, 4)
        refined_tiles.append(vis)

    self._axis_prior_visualization_count += 1
    record_dir = os.path.join(
        self.axis_prior_visualization_dir,
      self._axis_prior_visualization_run_id,
        f"register_{self._axis_prior_visualization_count:04d}",
    )
    os.makedirs(record_dir, exist_ok=True)
    images = {
        "input_rgb.jpg": color,
        "candidates_before.jpg": make_contact_sheet(top_indices),
        "candidates_boundary.jpg": make_contact_sheet(boundary_indices),
        "candidates_kept.jpg": make_contact_sheet(kept_ranked),
        "candidates_kept_after_refiner.jpg": make_image_contact_sheet(refined_tiles),
    }
    for filename, image in images.items():
      path = os.path.join(record_dir, filename)
      if not cv2.imwrite(path, np.ascontiguousarray(image[..., ::-1])):
        raise RuntimeError(f"cv2.imwrite failed: {path}")

    summary = {
        "candidates_before": int(len(poses)),
        "candidates_after": int(kept_count),
        "max_angle_deg": float(diagnostics["threshold_deg"]),
        "visualized_top_n": int(len(top_indices)),
        "boundary_rank_start": int(boundary_start + 1),
        "boundary_rank_end": int(boundary_end),
        "candidates": [
            {
                "candidate_index": int(index),
                "rank": int(rank_by_index[int(index)]),
                "angle_deg": float(diagnostics["angles_deg"][index]),
                "alignment": float(diagnostics["alignment"][index]),
                "angle_passed": bool(diagnostics["angle_pass_mask"][index]),
                "kept": int(index) in kept_set,
                "rendered_mask_iou": iou_by_index.get(int(index)),
            }
            for index in ranked
        ],
    }
    with open(os.path.join(record_dir, "summary.json"), "w", encoding="utf-8") as file:
      json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"[AxisPriorDebug] Saved candidate visualization: {record_dir}")

  def mask_iou(self, K: np.ndarray, image_shape: tuple[int, int], target_mask: np.ndarray, rendered_mask: np.ndarray | None = None) -> float:
    if not self.initialized:
      return 0.0
    if rendered_mask is None:
      rendered_mask = self.render_pose_mask(K, image_shape)
    rendered_mask = np.ascontiguousarray(rendered_mask > 0, dtype=np.uint8)
    target = np.ascontiguousarray(target_mask > 0, dtype=np.uint8)
    intersection = np.logical_and(rendered_mask > 0, target > 0).sum()
    union = np.logical_or(rendered_mask > 0, target > 0).sum()
    return float(intersection / union) if union > 0 else 0.0

  def draw_visualization(
      self,
      color: np.ndarray,
      K: np.ndarray,
      pose: np.ndarray | None = None,
      centered_pose: np.ndarray | None = None,
      rendered_mask: np.ndarray | None = None,
  ) -> np.ndarray:
    color = np.ascontiguousarray(color, dtype=np.uint8)
    K = np.ascontiguousarray(K, dtype=np.float32)
    if pose is None:
      pose = self.last_pose
    if pose is None:
      return color.copy()

    center_pose = pose @ np.linalg.inv(self.to_origin)
    vis = color.copy()
    if self.vis_mode in ("box", "both"):
      vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=self.bbox)
    if self.vis_mode in ("contour", "both"):
      if rendered_mask is None:
        rendered_mask = self.render_pose_mask(K, color.shape[:2], pose=centered_pose)
      rendered_mask = np.asarray(rendered_mask)
      if rendered_mask.shape != color.shape[:2]:
        raise ValueError(f"rendered mask/image shape mismatch: mask={rendered_mask.shape}, image={color.shape[:2]}")
      contour_mask = np.ascontiguousarray(rendered_mask > 0, dtype=np.uint8) * 255
      contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      if contours:
        cv2.drawContours(vis, contours, -1, color=(255, 255, 0), thickness=self.contour_thickness, lineType=cv2.LINE_AA)
    vis = draw_xyz_axis(
        vis,
        ob_in_cam=center_pose,
        scale=self.axis_scale,
        K=K,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    return vis

  @staticmethod
  def _valid_mask(mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    mask = np.ascontiguousarray(mask > 0, dtype=np.uint8)
    mask = mask & (depth >= 0.001)
    return np.ascontiguousarray(mask, dtype=np.uint8)

  @staticmethod
  def _prepare_frame_inputs(color: np.ndarray, depth: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    color = np.ascontiguousarray(color, dtype=np.uint8)
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    K = np.ascontiguousarray(K, dtype=np.float32)
    return color, depth, K

  def _ensure_estimator_float32(self) -> None:
    self.estimator.mesh.vertices = np.asarray(self.estimator.mesh.vertices, dtype=np.float32)
    self.estimator.pts = self.estimator.pts.float().contiguous()
    self.estimator.normals = self.estimator.normals.float().contiguous()
    self.estimator.rot_grid = self.estimator.rot_grid.float().contiguous()
    for key, value in self.estimator.mesh_tensors.items():
      if torch.is_tensor(value) and value.is_floating_point():
        self.estimator.mesh_tensors[key] = value.float().contiguous()