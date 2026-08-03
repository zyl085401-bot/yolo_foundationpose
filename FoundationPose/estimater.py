# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
import time


def merge_timing_details(*details):
  """Merge numeric timings without treating backend metadata or N/A values as seconds."""
  keys = set().union(*(detail.keys() for detail in details))
  merged = {}
  for key in keys:
    values = [detail.get(key) for detail in details]
    non_null_values = [value for value in values if value is not None]
    if key == 'network_backend':
      unique_values = list(dict.fromkeys(non_null_values))
      merged[key] = unique_values[0] if len(unique_values) == 1 else 'mixed'
    elif key == 'fallback_reason':
      unique_values = list(dict.fromkeys(value for value in non_null_values if value))
      merged[key] = '; '.join(unique_values) or None
    elif not non_null_values:
      merged[key] = None
    elif all(isinstance(value, (int, float, np.number)) for value in non_null_values):
      merged[key] = sum(float(value) for value in non_null_values)
    else:
      unique_values = list(dict.fromkeys(str(value) for value in non_null_values))
      merged[key] = unique_values[0] if len(unique_values) == 1 else '; '.join(unique_values)
  return merged


class FoundationPose:
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/', init_min_n_views=40, init_inplane_step=60):
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(min_n_views=init_min_n_views, inplane_step=init_inplane_step)

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    min_xyz = self.min_xyz
    max_xyz = self.max_xyz
    self.geometry_bbox_corners = torch.as_tensor([
      [min_xyz[0], min_xyz[1], min_xyz[2]],
      [min_xyz[0], min_xyz[1], max_xyz[2]],
      [min_xyz[0], max_xyz[1], min_xyz[2]],
      [min_xyz[0], max_xyz[1], max_xyz[2]],
      [max_xyz[0], min_xyz[1], min_xyz[2]],
      [max_xyz[0], min_xyz[1], max_xyz[2]],
      [max_xyz[0], max_xyz[1], min_xyz[2]],
      [max_xyz[0], max_xyz[1], max_xyz[2]],
    ], device='cuda', dtype=torch.float)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)
    self.render_lod_mesh = None
    self.render_lod_mesh_tensors = None
    self.render_lod_stages = set()

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")


  def configure_render_lod(self, mesh, stages):
    if mesh is None:
      raise ValueError('LOD render mesh is required')
    stages = {str(stage) for stage in stages}
    if not stages:
      raise ValueError('At least one LOD render stage must be enabled')
    self.render_lod_mesh = mesh
    self.render_lod_mesh_tensors = make_mesh_tensors(mesh)
    self.render_lod_stages = stages


  def clear_render_lod(self):
    self.render_lod_mesh = None
    self.render_lod_mesh_tensors = None
    self.render_lod_stages = set()


  def get_render_mesh_tensors(self, stage):
    if self.render_lod_mesh_tensors is not None and stage in self.render_lod_stages:
      return self.render_lod_mesh_tensors
    return self.mesh_tensors


  def get_render_mesh_source(self, stage):
    if self.render_lod_mesh_tensors is not None and stage in self.render_lod_stages:
      return 'lod'
    return 'original'



  def get_tf_to_centered_mesh(self):
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.render_lod_mesh_tensors is not None:
      for k in self.render_lod_mesh_tensors:
        logging.info(f"Moving LOD {k} to device {s}")
        self.render_lod_mesh_tensors[k] = self.render_lod_mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)



  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def prepare_frame_statistics(self, depth, mask):
    mask_positive = mask>0
    mask_rows, mask_cols = np.where(mask_positive)
    depth_valid = depth>=0.001
    valid_depth = mask.astype(bool) & depth_valid
    return {
      'mask_positive': mask_positive,
      'mask_rows': mask_rows,
      'mask_cols': mask_cols,
      'valid_positive_depth': mask_positive & depth_valid,
      'valid_depth': valid_depth,
      'median_depth': np.median(depth[valid_depth]) if valid_depth.any() else None,
    }


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None, frame_statistics=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K, frame_statistics=frame_statistics)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(self, depth, mask, K, frame_statistics=None):
    if frame_statistics is not None and 'translation_center' in frame_statistics:
      center = frame_statistics['translation_center']
      status = frame_statistics.get('translation_status')
      if status == 'empty_mask':
        logging.info(f'mask is all zero')
      elif status == 'empty_valid_depth':
        logging.info(f"valid is empty")
      if self.debug>=2 and status is None:
        pcd = toOpen3dCloud(center.reshape(1,3))
        o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)
      return center

    if frame_statistics is None:
      vs,us = np.where(mask>0)
      valid = mask.astype(bool) & (depth>=0.001)
      zc = np.median(depth[valid]) if valid.any() else None
    else:
      vs = frame_statistics['mask_rows']
      us = frame_statistics['mask_cols']
      valid = frame_statistics['valid_depth']
      zc = frame_statistics['median_depth']

    if len(us)==0:
      logging.info(f'mask is all zero')
      center = np.zeros((3))
      if frame_statistics is not None:
        frame_statistics['translation_center'] = center
        frame_statistics['translation_status'] = 'empty_mask'
      return center
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    if not valid.any():
      logging.info(f"valid is empty")
      center = np.zeros((3))
      if frame_statistics is not None:
        frame_statistics['translation_center'] = center
        frame_statistics['translation_status'] = 'empty_valid_depth'
      return center

    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc
    center = center.reshape(3)
    if frame_statistics is not None:
      frame_statistics['translation_center'] = center

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center


  def compute_geometry_candidate_score(self, poses, K, depth, mask, frame_statistics=None, return_components=False):
    if frame_statistics is None:
      vs, us = np.where(mask>0)
      valid = mask.astype(bool) & (depth>=0.001)
      median_depth = float(np.median(depth[valid])) if valid.any() else None
    else:
      vs = frame_statistics['mask_rows']
      us = frame_statistics['mask_cols']
      valid = frame_statistics['valid_depth']
      median_depth = frame_statistics['median_depth']
    if len(us)==0:
      score = torch.zeros(len(poses), device=poses.device, dtype=torch.float)
      if return_components:
        return score, {
          'center_error': score.clone(),
          'depth_error': score.clone(),
          'area_error': score.clone(),
          'aspect_error': score.clone(),
          'bbox_iou': score.clone(),
          'total_score': score,
        }
      return score

    if valid.any():
      median_depth = float(median_depth)
    else:
      median_depth = float(poses[:,2,3].median().detach().cpu())

    target_left = float(us.min())
    target_right = float(us.max())
    target_top = float(vs.min())
    target_bottom = float(vs.max())
    target_center_x = (target_left + target_right) / 2.0
    target_center_y = (target_top + target_bottom) / 2.0
    target_width = max(target_right - target_left, 1.0)
    target_height = max(target_bottom - target_top, 1.0)
    target_area = max(target_width * target_height, 1.0)
    target_aspect = target_width / target_height

    corners = self.geometry_bbox_corners.to(device=poses.device)

    cam_points = torch.einsum('bij,kj->bki', poses[:,:3,:3], corners) + poses[:,:3,3].reshape(-1,1,3)
    if frame_statistics is None:
      K_t = torch.as_tensor(K, device=poses.device, dtype=torch.float)
    else:
      K_t = frame_statistics.get('K_cuda')
      if K_t is None or K_t.device != poses.device:
        K_t = torch.as_tensor(K, device=poses.device, dtype=torch.float)
        frame_statistics['K_cuda'] = K_t
    projected = torch.einsum('ij,bkj->bki', K_t, cam_points)
    z = projected[...,2].clamp(min=1e-6)
    u = projected[...,0] / z
    v = projected[...,1] / z

    left = u.min(dim=1)[0]
    right = u.max(dim=1)[0]
    top = v.min(dim=1)[0]
    bottom = v.max(dim=1)[0]
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    width = (right - left).clamp(min=1.0)
    height = (bottom - top).clamp(min=1.0)
    area = (width * height).clamp(min=1.0)
    aspect = width / height

    inter_left = torch.maximum(left, torch.tensor(target_left, device=poses.device))
    inter_right = torch.minimum(right, torch.tensor(target_right, device=poses.device))
    inter_top = torch.maximum(top, torch.tensor(target_top, device=poses.device))
    inter_bottom = torch.minimum(bottom, torch.tensor(target_bottom, device=poses.device))
    inter_area = (inter_right - inter_left).clamp(min=0.0) * (inter_bottom - inter_top).clamp(min=0.0)
    union_area = area + target_area - inter_area
    bbox_iou = inter_area / union_area.clamp(min=1.0)

    center_error = ((center_x - target_center_x) / target_width).square() + ((center_y - target_center_y) / target_height).square()
    depth_error = ((poses[:,2,3] - median_depth) / max(self.diameter, 1e-6)).abs()
    area_error = torch.log(area / target_area).abs()
    aspect_error = torch.log(aspect / target_aspect).abs()
    score = center_error + depth_error + 0.5 * area_error + 0.5 * aspect_error + (1.0 - bbox_iou)
    if return_components:
      return score, {
        'center_error': center_error,
        'depth_error': depth_error,
        'area_error': area_error,
        'aspect_error': aspect_error,
        'bbox_iou': bbox_iou,
        'total_score': score,
      }
    return score


  def select_coarse_score_candidates_by_geometry(self, poses, K, depth, mask, top_k, frame_statistics=None):
    if top_k>=len(poses):
      return torch.arange(len(poses), device=poses.device)
    score = self.compute_geometry_candidate_score(poses, K, depth, mask, frame_statistics=frame_statistics)
    return score.argsort()[:top_k]


  def estimate_axis_prior_from_depth_pca(self, xyz_map, mask, min_points=500, min_confidence=1.4, max_points=3000, frame_statistics=None):
    if frame_statistics is None:
      valid = (mask>0) & (xyz_map[...,2]>=0.001) & np.isfinite(xyz_map).all(axis=-1)
      points = xyz_map[valid]
    else:
      points = xyz_map[frame_statistics['mask_positive']]
      valid = (points[:,2]>=0.001) & np.isfinite(points).all(axis=-1)
      points = points[valid]
    if len(points)<min_points:
      return None, {
        'axis_prior_status': 'too_few_points',
        'axis_prior_points': int(len(points)),
        'axis_prior_confidence': 0.0,
      }

    if len(points)>max_points:
      ids = np.linspace(0, len(points)-1, max_points).astype(np.int64)
      points = points[ids]

    points = points.astype(np.float32)
    centered = points - points.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered)-1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:,order]
    confidence = float(eigvals[0] / max(eigvals[1], 1e-8))
    if not np.isfinite(confidence) or confidence<min_confidence:
      return None, {
        'axis_prior_status': 'low_confidence',
        'axis_prior_points': int(len(points)),
        'axis_prior_confidence': confidence,
        'axis_prior_eigenvalues': [float(v) for v in eigvals],
      }

    axis = eigvecs[:,0].astype(np.float32)
    norm = np.linalg.norm(axis)
    if norm<1e-8:
      return None, {
        'axis_prior_status': 'degenerate_axis',
        'axis_prior_points': int(len(points)),
        'axis_prior_confidence': confidence,
        'axis_prior_eigenvalues': [float(v) for v in eigvals],
      }
    axis = axis / norm
    return axis, {
      'axis_prior_status': 'used',
      'axis_prior_points': int(len(points)),
      'axis_prior_confidence': confidence,
      'axis_prior_eigenvalues': [float(v) for v in eigvals],
      'axis_prior_scene_axis': [float(v) for v in axis],
    }


  def filter_pose_candidates_by_axis_prior(self, poses, scene_axis, model_axis=(0,0,1), max_angle_deg=45, min_candidates=12, max_candidates=0, collect_diagnostics=False, return_candidate_metrics=False):
    if scene_axis is None or len(poses)==0:
      filter_info = {
        'axis_prior_candidates_before': int(len(poses)),
        'axis_prior_candidates_after': int(len(poses)),
      }
      if return_candidate_metrics:
        return poses, filter_info, None
      return poses, filter_info

    model_axis = np.asarray(model_axis, dtype=np.float32).reshape(3)
    model_norm = np.linalg.norm(model_axis)
    if model_norm<1e-8:
      filter_info = {
        'axis_prior_status': 'invalid_model_axis',
        'axis_prior_candidates_before': int(len(poses)),
        'axis_prior_candidates_after': int(len(poses)),
      }
      if return_candidate_metrics:
        return poses, filter_info, None
      return poses, filter_info
    model_axis = model_axis / model_norm

    scene_axis = np.asarray(scene_axis, dtype=np.float32).reshape(3)
    scene_norm = np.linalg.norm(scene_axis)
    if scene_norm<1e-8:
      filter_info = {
        'axis_prior_status': 'invalid_scene_axis',
        'axis_prior_candidates_before': int(len(poses)),
        'axis_prior_candidates_after': int(len(poses)),
      }
      if return_candidate_metrics:
        return poses, filter_info, None
      return poses, filter_info
    scene_axis = scene_axis / scene_norm

    model_axis_t = torch.as_tensor(model_axis, device=poses.device, dtype=poses.dtype)
    scene_axis_t = torch.as_tensor(scene_axis, device=poses.device, dtype=poses.dtype)
    candidate_axes = torch.matmul(poses[:,:3,:3], model_axis_t)
    candidate_axes = F.normalize(candidate_axes, dim=-1)
    alignment = torch.abs((candidate_axes * scene_axis_t.reshape(1,3)).sum(dim=-1))
    threshold = math.cos(math.radians(float(max_angle_deg)))
    keep = torch.where(alignment>=threshold)[0]
    min_candidates = max(1, min(int(min_candidates), len(poses)))
    if len(keep)<min_candidates:
      keep = alignment.argsort(descending=True)[:min_candidates]
    max_candidates = int(max_candidates)
    if max_candidates>0 and len(keep)>max_candidates:
      max_candidates = max(min_candidates, min(max_candidates, len(poses)))
      keep = keep[alignment[keep].argsort(descending=True)[:max_candidates]]
    candidate_metrics = None
    if collect_diagnostics or return_candidate_metrics:
      ranked_indices = alignment.argsort(descending=True)
      kept_mask = torch.zeros(len(poses), device=poses.device, dtype=torch.bool)
      kept_mask[keep] = True
      angles_deg = torch.rad2deg(torch.acos(alignment.clamp(0, 1)))
      candidate_metrics = {
          'alignment': alignment.detach().cpu().numpy().astype(np.float32),
          'angles_deg': angles_deg.detach().cpu().numpy().astype(np.float32),
          'ranked_indices': ranked_indices.detach().cpu().numpy().astype(np.int64),
          'kept_indices': keep.detach().cpu().numpy().astype(np.int64),
          'kept_mask': kept_mask.detach().cpu().numpy(),
          'angle_pass_mask': (alignment>=threshold).detach().cpu().numpy(),
          'threshold_deg': float(max_angle_deg),
      }
    if collect_diagnostics:
      self.last_axis_prior_diagnostics = {
          'poses_before': poses.detach().cpu().numpy().astype(np.float32),
          **candidate_metrics,
      }
    poses_filtered = poses[keep]
    filter_info = {
      'axis_prior_candidates_before': int(len(poses)),
      'axis_prior_candidates_after': int(len(poses_filtered)),
      'axis_prior_max_candidates': int(max_candidates),
      'axis_prior_max_alignment': float(alignment.max().detach().cpu()),
      'axis_prior_min_kept_alignment': float(alignment[keep].min().detach().cpu()),
      'axis_prior_max_angle_deg': float(max_angle_deg),
      'axis_prior_model_axis': [float(v) for v in model_axis],
    }
    if return_candidate_metrics:
      return poses_filtered, filter_info, candidate_metrics
    return poses_filtered, filter_info


  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, init_strategy='default', coarse_refine_iter=1, coarse_score_filter='none', coarse_score_top_k=999999, fine_refine_iter=2, fine_top_k=16, axis_prior_filter='none', axis_prior_model_axis=(0,0,1), axis_prior_max_angle_deg=45, axis_prior_min_candidates=12, axis_prior_max_candidates=0, axis_prior_min_points=500, axis_prior_min_confidence=1.4, axis_prior_debug=False, skip_redundant_coarse_scorer=False, frame_statistics_reuse_enabled=False, candidate_pipeline_debug_enabled=False, distillation_capture_enabled=False):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    timing = {}
    self.last_axis_prior_diagnostics = None
    self.last_candidate_diagnostics = None
    self.last_distillation_group = None
    candidate_identity_enabled = bool(candidate_pipeline_debug_enabled or distillation_capture_enabled)
    candidate_diagnostics = None
    if candidate_pipeline_debug_enabled:
      candidate_diagnostics = {
        'version': 1,
        'init_strategy': str(init_strategy),
        'stages': [],
      }
      self.last_candidate_diagnostics = candidate_diagnostics

    def capture_candidate_stage(name, poses_value, candidate_ids_value, parent_stage=None, selected_candidate_ids=None, **metrics):
      if candidate_diagnostics is None:
        return
      poses_numpy = poses_value.detach().cpu().numpy() if torch.is_tensor(poses_value) else np.asarray(poses_value)
      candidate_ids_numpy = np.asarray(candidate_ids_value, dtype=np.int64).reshape(-1)
      stage = {
        'name': str(name),
        'parent_stage': parent_stage,
        'poses': np.asarray(poses_numpy, dtype=np.float32).reshape(-1, 4, 4),
        'candidate_ids': candidate_ids_numpy,
      }
      if selected_candidate_ids is not None:
        selected_ids_numpy = np.asarray(selected_candidate_ids, dtype=np.int64).reshape(-1)
        stage['selected_candidate_ids'] = selected_ids_numpy
        stage['selected_for_next'] = np.isin(candidate_ids_numpy, selected_ids_numpy)
      for key, value in metrics.items():
        if value is not None:
          stage[key] = value
      candidate_diagnostics['stages'].append(stage)

    t_register_start = time.perf_counter()
    set_seed(0)
    # Registration repeatedly uses fixed 160x160 inputs and stable batch sizes,
    # so cache cuDNN's fastest deterministic convolution algorithms per shape.
    torch.backends.cudnn.benchmark = True
    logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    t0 = time.perf_counter()
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    torch.cuda.synchronize()
    timing['depth_preprocess'] = time.perf_counter() - t0

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    frame_statistics = None
    timing['frame_statistics_reuse_status'] = 'enabled' if frame_statistics_reuse_enabled else 'disabled'
    if frame_statistics_reuse_enabled:
      t0 = time.perf_counter()
      frame_statistics = self.prepare_frame_statistics(depth=depth, mask=ob_mask)
      timing['frame_statistics'] = time.perf_counter() - t0
      valid = frame_statistics['valid_positive_depth']
    else:
      timing['frame_statistics'] = 0.0
      valid = (depth>=0.001) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K, frame_statistics=frame_statistics)
      torch.cuda.synchronize()
      timing['register'] = time.perf_counter() - t_register_start
      self.last_register_timing = timing
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    t0 = time.perf_counter()
    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None, frame_statistics=frame_statistics)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K, frame_statistics=frame_statistics)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')
    candidate_ids = np.arange(len(poses), dtype=np.int64)
    initial_poses = poses.detach().cpu().numpy().astype(np.float32) if candidate_pipeline_debug_enabled else None
    timing['pose_hypothesis_candidates'] = len(poses)

    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")
    torch.cuda.synchronize()
    timing['pose_hypothesis'] = time.perf_counter() - t0

    xyz_map = depth2xyzmap(depth, K)
    timing['axis_prior_filter'] = axis_prior_filter
    timing['axis_prior_model_axis'] = [float(v) for v in axis_prior_model_axis]
    timing['axis_prior_max_angle_deg'] = float(axis_prior_max_angle_deg)
    timing['axis_prior_min_candidates'] = int(axis_prior_min_candidates)
    timing['axis_prior_max_candidates_config'] = int(axis_prior_max_candidates)
    timing['axis_prior_min_confidence'] = float(axis_prior_min_confidence)
    timing['axis_prior_candidates_before'] = int(len(poses))
    timing['axis_prior_candidates_after'] = int(len(poses))
    axis_candidate_metrics = None
    if axis_prior_filter == 'depth_pca':
      t0 = time.perf_counter()
      scene_axis, axis_info = self.estimate_axis_prior_from_depth_pca(
          xyz_map=xyz_map,
          mask=ob_mask,
          min_points=axis_prior_min_points,
          min_confidence=axis_prior_min_confidence,
          frame_statistics=frame_statistics,
      )
      filter_result = self.filter_pose_candidates_by_axis_prior(
          poses=poses,
          scene_axis=scene_axis,
          model_axis=axis_prior_model_axis,
          max_angle_deg=axis_prior_max_angle_deg,
          min_candidates=axis_prior_min_candidates,
          max_candidates=axis_prior_max_candidates,
          collect_diagnostics=axis_prior_debug,
          return_candidate_metrics=candidate_identity_enabled,
      )
      if candidate_identity_enabled:
        poses, filter_info, axis_candidate_metrics = filter_result
        if axis_candidate_metrics is not None:
          kept_positions = np.asarray(axis_candidate_metrics['kept_indices'], dtype=np.int64)
          selected_candidate_ids = candidate_ids[kept_positions]
          axis_rank = np.empty(len(candidate_ids), dtype=np.int64)
          axis_rank[np.asarray(axis_candidate_metrics['ranked_indices'], dtype=np.int64)] = np.arange(1, len(candidate_ids) + 1, dtype=np.int64)
        else:
          selected_candidate_ids = candidate_ids.copy()
          axis_rank = None
        if candidate_pipeline_debug_enabled:
          capture_candidate_stage(
              'initial_hypotheses',
              initial_poses,
              candidate_ids,
              selected_candidate_ids=selected_candidate_ids,
              axis_alignment=None if axis_candidate_metrics is None else axis_candidate_metrics['alignment'],
              axis_angle_deg=None if axis_candidate_metrics is None else axis_candidate_metrics['angles_deg'],
              axis_rank=axis_rank,
              axis_ranked_positions=None if axis_candidate_metrics is None else axis_candidate_metrics['ranked_indices'],
              axis_angle_pass=None if axis_candidate_metrics is None else axis_candidate_metrics['angle_pass_mask'],
          )
        candidate_ids = selected_candidate_ids
      else:
        poses, filter_info = filter_result
      torch.cuda.synchronize()
      timing['axis_prior'] = time.perf_counter() - t0
      timing.update(axis_info)
      timing.update(filter_info)
      timing['pose_hypothesis_candidates_after_axis_prior'] = len(poses)
    else:
      timing['axis_prior_status'] = 'disabled'
      if candidate_pipeline_debug_enabled:
        capture_candidate_stage(
            'initial_hypotheses',
            initial_poses,
            candidate_ids,
            selected_candidate_ids=candidate_ids,
        )

    if candidate_pipeline_debug_enabled:
      capture_candidate_stage(
          'axis_prior_kept',
          poses,
          candidate_ids,
          parent_stage='initial_hypotheses',
          selected_candidate_ids=candidate_ids,
      )
      candidate_diagnostics['axis_prior'] = {
        key: timing[key]
        for key in timing
        if key.startswith('axis_prior_')
      }

    refiner_stage1_enabled = getattr(self.refiner, 'refiner_stage1_optimizations_enabled', False)
    if refiner_stage1_enabled:
      t0 = time.perf_counter()
      rgb_cuda = torch.as_tensor(rgb, dtype=torch.float, device='cuda')
      depth_cuda = torch.as_tensor(depth, dtype=torch.float, device='cuda')
      xyz_map_cuda = torch.as_tensor(xyz_map, dtype=torch.float, device='cuda')
      torch.cuda.synchronize()
      timing['frame_to_cuda'] = time.perf_counter() - t0
    else:
      rgb_cuda = rgb
      depth_cuda = depth
      xyz_map_cuda = xyz_map
    scorer_xyz_map = None

    if init_strategy == 'topk_two_stage':
      timing['refiner_coarse_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      refiner_pose_input = poses if refiner_stage1_enabled else poses.data.cpu().numpy()
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.get_render_mesh_tensors('refiner_coarse'), rgb=rgb_cuda, depth=depth_cuda, K=K, ob_in_cams=refiner_pose_input, normal_map=normal_map, xyz_map=xyz_map_cuda, glctx=self.glctx, mesh_diameter=self.diameter, iteration=coarse_refine_iter, get_vis=False, network_stage='refiner_coarse', capture_iteration_poses=candidate_pipeline_debug_enabled)
      torch.cuda.synchronize()
      coarse_parent_stage = 'axis_prior_kept'
      if candidate_pipeline_debug_enabled:
        for iteration_index, iteration_poses in enumerate(getattr(self.refiner, 'last_iteration_poses', ()), start=1):
          stage_name = f'coarse_refiner_iter_{iteration_index}'
          capture_candidate_stage(
              stage_name,
              iteration_poses,
              candidate_ids,
              parent_stage=coarse_parent_stage,
              selected_candidate_ids=candidate_ids,
          )
          coarse_parent_stage = stage_name
      if self.last_axis_prior_diagnostics is not None:
        self.last_axis_prior_diagnostics['poses_after_coarse_refiner'] = poses.detach().cpu().numpy().astype(np.float32)
      timing['refiner_coarse'] = time.perf_counter() - t0
      timing['refiner_coarse_detail'] = dict(getattr(self.refiner, 'last_timing', {}))

      t0 = time.perf_counter()
      score_k = max(1, min(int(coarse_score_top_k), len(poses)))
      geometry_components = None
      if coarse_score_filter == 'geometry' and candidate_pipeline_debug_enabled:
        geometry_scores, geometry_components = self.compute_geometry_candidate_score(
            poses,
            K,
            depth,
            ob_mask,
            frame_statistics=frame_statistics,
            return_components=True,
        )
        if score_k < len(poses):
          score_ids = geometry_scores.argsort()[:score_k]
        else:
          score_ids = torch.linspace(0, len(poses) - 1, steps=score_k, device=poses.device).long()
      elif coarse_score_filter == 'geometry' and score_k < len(poses):
        score_ids = self.select_coarse_score_candidates_by_geometry(poses, K, depth, ob_mask, score_k, frame_statistics=frame_statistics)
      else:
        score_ids = torch.linspace(0, len(poses) - 1, steps=score_k, device=poses.device).long()
      score_poses = poses[score_ids]
      if candidate_identity_enabled:
        score_positions = score_ids.detach().cpu().numpy().astype(np.int64)
        score_candidate_ids = candidate_ids[score_positions]
        geometry_metrics = {}
        if candidate_pipeline_debug_enabled and geometry_components is not None:
          for key, value in geometry_components.items():
            geometry_metrics[f'geometry_{key}'] = value.detach().cpu().numpy().astype(np.float32)
          geometry_order = geometry_metrics['geometry_total_score'].argsort()
          geometry_rank = np.empty(len(geometry_order), dtype=np.int64)
          geometry_rank[geometry_order] = np.arange(1, len(geometry_order) + 1, dtype=np.int64)
          geometry_metrics['geometry_rank'] = geometry_rank
        if candidate_pipeline_debug_enabled:
          capture_candidate_stage(
              'geometry_filter',
              poses,
              candidate_ids,
              parent_stage=coarse_parent_stage,
              selected_candidate_ids=score_candidate_ids,
              filter_mode=str(coarse_score_filter),
              **geometry_metrics,
          )
      if self.last_axis_prior_diagnostics is not None:
        self.last_axis_prior_diagnostics['coarse_selected_positions'] = score_ids.detach().cpu().numpy().astype(np.int64)
      torch.cuda.synchronize()
      timing['coarse_score_select'] = time.perf_counter() - t0
      timing['coarse_score_candidates'] = score_k
      timing['scorer_coarse_candidates'] = int(len(score_poses))
      timing['coarse_score_filter'] = coarse_score_filter
      top_k = max(1, min(int(fine_top_k), len(poses)))
      top_k = min(top_k, len(score_poses))
      coarse_scorer_redundant = (
          bool(skip_redundant_coarse_scorer)
          and top_k == len(score_poses)
          and self.last_axis_prior_diagnostics is None
      )

      if coarse_scorer_redundant:
        timing['scorer_coarse'] = 0.0
        timing['scorer_coarse_detail'] = {}
        timing['coarse_scorer_status'] = 'skipped_redundant_all_candidates_retained'
        coarse_scores_numpy = None
      else:
        t0 = time.perf_counter()
        if getattr(self.scorer, 'scorer_precomputed_xyz_enabled', False) and scorer_xyz_map is None:
          scorer_xyz_map = xyz_map_cuda if refiner_stage1_enabled else torch.as_tensor(xyz_map, dtype=torch.float, device='cuda')
        xyz_map_for_scorer = scorer_xyz_map if scorer_xyz_map is not None else xyz_map
        scorer_pose_input = score_poses if refiner_stage1_enabled else score_poses.data.cpu().numpy()
        scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb_cuda, depth=depth_cuda, K=K, ob_in_cams=scorer_pose_input, normal_map=normal_map, xyz_map=xyz_map_for_scorer, mesh_tensors=self.get_render_mesh_tensors('scorer_coarse'), glctx=self.glctx, mesh_diameter=self.diameter, get_vis=False, network_stage='scorer_coarse')
        torch.cuda.synchronize()
        coarse_scores_numpy = None
        if candidate_pipeline_debug_enabled:
          coarse_scores_numpy = np.asarray(scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores, dtype=np.float32).reshape(-1)
        timing['scorer_coarse_detail'] = dict(getattr(self.scorer, 'last_timing', {}))
        if self.last_axis_prior_diagnostics is not None:
          coarse_scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
          self.last_axis_prior_diagnostics['coarse_scores'] = np.asarray(coarse_scores, dtype=np.float32).reshape(-1)
        timing['scorer_coarse'] = time.perf_counter() - t0
        timing['coarse_scorer_status'] = 'executed'

      t0 = time.perf_counter()
      if coarse_scorer_redundant:
        poses = score_poses
        if candidate_identity_enabled:
          selected_candidate_ids = score_candidate_ids.copy()
          coarse_rank = None
      else:
        top_ids = torch.as_tensor(scores).argsort(descending=True)[:top_k]
        poses = score_poses[top_ids]
        if candidate_identity_enabled:
          top_positions = top_ids.detach().cpu().numpy().astype(np.int64)
          selected_candidate_ids = score_candidate_ids[top_positions]
          if candidate_pipeline_debug_enabled:
            coarse_order = coarse_scores_numpy.argsort()[::-1]
            coarse_rank = np.empty(len(coarse_order), dtype=np.int64)
            coarse_rank[coarse_order] = np.arange(1, len(coarse_order) + 1, dtype=np.int64)
      if candidate_pipeline_debug_enabled:
        capture_candidate_stage(
            'coarse_scorer',
            score_poses,
            score_candidate_ids,
            parent_stage='geometry_filter',
            selected_candidate_ids=selected_candidate_ids,
            scorer_status=timing['coarse_scorer_status'],
            scorer_score=coarse_scores_numpy,
            scorer_rank=coarse_rank,
        )
      if candidate_identity_enabled:
        candidate_ids = selected_candidate_ids
      torch.cuda.synchronize()
      timing['topk_select'] = time.perf_counter() - t0

      timing['refiner_fine_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      refiner_pose_input = poses if refiner_stage1_enabled else poses.data.cpu().numpy()
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.get_render_mesh_tensors('refiner_fine'), rgb=rgb_cuda, depth=depth_cuda, K=K, ob_in_cams=refiner_pose_input, normal_map=normal_map, xyz_map=xyz_map_cuda, glctx=self.glctx, mesh_diameter=self.diameter, iteration=fine_refine_iter, get_vis=self.debug>=2, network_stage='refiner_fine', capture_iteration_poses=candidate_pipeline_debug_enabled, capture_distillation_data=distillation_capture_enabled)
      torch.cuda.synchronize()
      fine_parent_stage = 'coarse_scorer'
      if candidate_pipeline_debug_enabled:
        for iteration_index, iteration_poses in enumerate(getattr(self.refiner, 'last_iteration_poses', ()), start=1):
          stage_name = f'fine_refiner_iter_{iteration_index}'
          capture_candidate_stage(
              stage_name,
              iteration_poses,
              candidate_ids,
              parent_stage=fine_parent_stage,
              selected_candidate_ids=candidate_ids,
          )
          fine_parent_stage = stage_name
      timing['refiner_fine'] = time.perf_counter() - t0
      timing['refiner_fine_detail'] = dict(getattr(self.refiner, 'last_timing', {}))
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

      timing['scorer_fine_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      if getattr(self.scorer, 'scorer_precomputed_xyz_enabled', False) and scorer_xyz_map is None:
        scorer_xyz_map = xyz_map_cuda if refiner_stage1_enabled else torch.as_tensor(xyz_map, dtype=torch.float, device='cuda')
      xyz_map_for_scorer = scorer_xyz_map if scorer_xyz_map is not None else xyz_map
      scorer_pose_input = poses if refiner_stage1_enabled else poses.data.cpu().numpy()
      scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb_cuda, depth=depth_cuda, K=K, ob_in_cams=scorer_pose_input, normal_map=normal_map, xyz_map=xyz_map_for_scorer, mesh_tensors=self.get_render_mesh_tensors('scorer_fine'), glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2, network_stage='scorer_fine')
      torch.cuda.synchronize()
      if distillation_capture_enabled:
        raw_teacher_logits = np.asarray(self.scorer.last_raw_score_logits, dtype=np.float32).reshape(-1)
        public_teacher_scores = np.asarray(scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores, dtype=np.float32).reshape(-1)
        refiner_capture = getattr(self.refiner, 'last_distillation_data', None)
        if refiner_capture is None:
          raise RuntimeError('Fine Refiner did not produce distillation capture data')
        if len(candidate_ids) != len(raw_teacher_logits):
          raise RuntimeError(
              f'Distillation candidate/logit mismatch: candidates={len(candidate_ids)}, logits={len(raw_teacher_logits)}'
          )
        if not np.isfinite(raw_teacher_logits).all():
          raise RuntimeError('Final Scorer raw logits are incomplete or non-finite')
        teacher_order_positions = raw_teacher_logits.argsort()[::-1].astype(np.int64)
        teacher_margin = (
            float(raw_teacher_logits[teacher_order_positions[0]] - raw_teacher_logits[teacher_order_positions[1]])
            if len(teacher_order_positions) > 1 else float('inf')
        )
        candidate_ids_array = np.asarray(candidate_ids, dtype=np.int64)
        self.last_distillation_group = {
            'schema_version': 1,
            'feature_version': str(refiner_capture['feature_version']),
            'candidate_ids': candidate_ids_array,
            'candidate_count': int(len(candidate_ids_array)),
            'fine_iterations': refiner_capture['iterations'],
            'teacher_raw_logits': raw_teacher_logits,
            'teacher_public_scores': public_teacher_scores,
            'teacher_order_positions': teacher_order_positions,
            'teacher_order_candidate_ids': candidate_ids_array[teacher_order_positions],
            'teacher_top1_candidate_id': int(candidate_ids_array[teacher_order_positions[0]]),
            'teacher_margin': teacher_margin,
            'teacher_backend': str(getattr(self.scorer, 'last_score_backend', None)),
            'final_poses': poses.detach().cpu().numpy().astype(np.float32),
        }
      if candidate_pipeline_debug_enabled:
        final_scores_numpy = np.asarray(scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores, dtype=np.float32).reshape(-1)
        final_order = final_scores_numpy.argsort()[::-1]
        final_rank = np.empty(len(final_order), dtype=np.int64)
        final_rank[final_order] = np.arange(1, len(final_order) + 1, dtype=np.int64)
        top1_candidate_id = int(candidate_ids[final_order[0]])
        capture_candidate_stage(
            'final_scorer',
            poses,
            candidate_ids,
            parent_stage=fine_parent_stage,
            selected_candidate_ids=np.asarray([top1_candidate_id], dtype=np.int64),
            scorer_score=final_scores_numpy,
            scorer_rank=final_rank,
        )
        candidate_diagnostics['top1_candidate_id'] = top1_candidate_id
        candidate_diagnostics['coarse_scorer_status'] = timing['coarse_scorer_status']
      timing['scorer_fine'] = time.perf_counter() - t0
      timing['scorer_fine_detail'] = dict(getattr(self.scorer, 'last_timing', {}))
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

      timing['refiner'] = timing['refiner_coarse'] + timing['refiner_fine']
      timing['scorer'] = timing['scorer_coarse'] + timing['scorer_fine']
      timing['scorer_detail'] = merge_timing_details(timing['scorer_coarse_detail'], timing['scorer_fine_detail'])
      timing['refiner_detail'] = merge_timing_details(timing['refiner_coarse_detail'], timing['refiner_fine_detail'])
    else:
      timing['refiner_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      refiner_pose_input = poses if refiner_stage1_enabled else poses.data.cpu().numpy()
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.get_render_mesh_tensors('refiner_default'), rgb=rgb_cuda, depth=depth_cuda, K=K, ob_in_cams=refiner_pose_input, normal_map=normal_map, xyz_map=xyz_map_cuda, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
      torch.cuda.synchronize()
      timing['refiner'] = time.perf_counter() - t0
      timing['refiner_detail'] = getattr(self.refiner, 'last_timing', {})
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

      timing['scorer_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      if getattr(self.scorer, 'scorer_precomputed_xyz_enabled', False) and scorer_xyz_map is None:
        scorer_xyz_map = xyz_map_cuda if refiner_stage1_enabled else torch.as_tensor(xyz_map, dtype=torch.float, device='cuda')
      xyz_map_for_scorer = scorer_xyz_map if scorer_xyz_map is not None else xyz_map
      scorer_pose_input = poses if refiner_stage1_enabled else poses.data.cpu().numpy()
      scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb_cuda, depth=depth_cuda, K=K, ob_in_cams=scorer_pose_input, normal_map=normal_map, xyz_map=xyz_map_for_scorer, mesh_tensors=self.get_render_mesh_tensors('scorer_default'), glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
      torch.cuda.synchronize()
      timing['scorer'] = time.perf_counter() - t0
      timing['scorer_detail'] = dict(getattr(self.scorer, 'last_timing', {}))
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    t0 = time.perf_counter()
    add_errs = self.compute_add_err_to_gt_pose(poses)
    logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]

    logging.info(f'sorted scores:{scores}')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]

    self.poses = poses
    self.scores = scores
    if candidate_pipeline_debug_enabled and init_strategy == 'topk_two_stage':
      sorted_positions = ids.detach().cpu().numpy().astype(np.int64)
      candidate_diagnostics['final_ranked_candidate_ids'] = candidate_ids[sorted_positions]
      candidate_diagnostics['final_ranked_centered_poses'] = poses.detach().cpu().numpy().astype(np.float32)
      candidate_diagnostics['final_ranked_scores'] = np.asarray(scores.detach().cpu().numpy() if torch.is_tensor(scores) else scores, dtype=np.float32).reshape(-1)
    torch.cuda.synchronize()
    timing['sort_select'] = time.perf_counter() - t0

    torch.cuda.synchronize()
    timing['register'] = time.perf_counter() - t_register_start
    known_time = sum(timing.get(key, 0.0) for key in ('depth_preprocess', 'frame_statistics', 'pose_hypothesis', 'axis_prior', 'frame_to_cuda', 'refiner', 'coarse_score_select', 'scorer', 'topk_select', 'sort_select'))
    timing['other'] = max(timing['register'] - known_time, 0.0)
    self.last_register_timing = timing

    return best_pose.data.cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    '''
    @poses: wrt. the centered mesh
    '''
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}):
    t_track_start = time.perf_counter()
    if self.pose_last is None:
      logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    track_pose_input = self.pose_last.reshape(1,4,4)
    if not getattr(self.refiner, 'refiner_stage1_optimizations_enabled', False):
      track_pose_input = track_pose_input.data.cpu().numpy()
    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.get_render_mesh_tensors('refiner_track'), rgb=rgb, depth=depth, K=K, ob_in_cams=track_pose_input, normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2, network_stage='refiner_track')
    torch.cuda.synchronize()
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    self.last_track_timing = {'track': time.perf_counter() - t_track_start}
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)


