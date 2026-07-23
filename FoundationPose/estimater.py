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
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      self.mesh_path = f'/tmp/{uuid.uuid4()}.obj'
      self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")



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


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):
    '''
    @scene_pts: torch tensor (N,3)
    '''
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(self, depth, mask, K):
    vs,us = np.where(mask>0)
    if len(us)==0:
      logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.001)
    if not valid.any():
      logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)


  def compute_geometry_candidate_score(self, poses, K, depth, mask):
    vs, us = np.where(mask>0)
    if len(us)==0:
      return torch.zeros(len(poses), device=poses.device, dtype=torch.float)

    valid = mask.astype(bool) & (depth>=0.001)
    if valid.any():
      median_depth = float(np.median(depth[valid]))
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

    min_xyz = torch.as_tensor(self.min_xyz, device=poses.device, dtype=torch.float)
    max_xyz = torch.as_tensor(self.max_xyz, device=poses.device, dtype=torch.float)
    corners = torch.stack([
      torch.stack([min_xyz[0], min_xyz[1], min_xyz[2]]),
      torch.stack([min_xyz[0], min_xyz[1], max_xyz[2]]),
      torch.stack([min_xyz[0], max_xyz[1], min_xyz[2]]),
      torch.stack([min_xyz[0], max_xyz[1], max_xyz[2]]),
      torch.stack([max_xyz[0], min_xyz[1], min_xyz[2]]),
      torch.stack([max_xyz[0], min_xyz[1], max_xyz[2]]),
      torch.stack([max_xyz[0], max_xyz[1], min_xyz[2]]),
      torch.stack([max_xyz[0], max_xyz[1], max_xyz[2]]),
    ], dim=0)

    cam_points = torch.einsum('bij,kj->bki', poses[:,:3,:3], corners) + poses[:,:3,3].reshape(-1,1,3)
    K_t = torch.as_tensor(K, device=poses.device, dtype=torch.float)
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
    return center_error + depth_error + 0.5 * area_error + 0.5 * aspect_error + (1.0 - bbox_iou)


  def select_coarse_score_candidates_by_geometry(self, poses, K, depth, mask, top_k):
    if top_k>=len(poses):
      return torch.arange(len(poses), device=poses.device)
    score = self.compute_geometry_candidate_score(poses, K, depth, mask)
    return score.argsort()[:top_k]


  def estimate_axis_prior_from_depth_pca(self, xyz_map, mask, min_points=500, min_confidence=1.4, max_points=3000):
    valid = (mask>0) & (xyz_map[...,2]>=0.001) & np.isfinite(xyz_map).all(axis=-1)
    points = xyz_map[valid]
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


  def filter_pose_candidates_by_axis_prior(self, poses, scene_axis, model_axis=(0,0,1), max_angle_deg=45, min_candidates=12, max_candidates=0, collect_diagnostics=False):
    if scene_axis is None or len(poses)==0:
      return poses, {
        'axis_prior_candidates_before': int(len(poses)),
        'axis_prior_candidates_after': int(len(poses)),
      }

    model_axis = np.asarray(model_axis, dtype=np.float32).reshape(3)
    model_norm = np.linalg.norm(model_axis)
    if model_norm<1e-8:
      return poses, {
        'axis_prior_status': 'invalid_model_axis',
        'axis_prior_candidates_before': int(len(poses)),
        'axis_prior_candidates_after': int(len(poses)),
      }
    model_axis = model_axis / model_norm

    scene_axis = np.asarray(scene_axis, dtype=np.float32).reshape(3)
    scene_norm = np.linalg.norm(scene_axis)
    if scene_norm<1e-8:
      return poses, {
        'axis_prior_status': 'invalid_scene_axis',
        'axis_prior_candidates_before': int(len(poses)),
        'axis_prior_candidates_after': int(len(poses)),
      }
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
    if collect_diagnostics:
      ranked_indices = alignment.argsort(descending=True)
      kept_mask = torch.zeros(len(poses), device=poses.device, dtype=torch.bool)
      kept_mask[keep] = True
      angles_deg = torch.rad2deg(torch.acos(alignment.clamp(0, 1)))
      self.last_axis_prior_diagnostics = {
          'poses_before': poses.detach().cpu().numpy().astype(np.float32),
          'alignment': alignment.detach().cpu().numpy().astype(np.float32),
          'angles_deg': angles_deg.detach().cpu().numpy().astype(np.float32),
          'ranked_indices': ranked_indices.detach().cpu().numpy().astype(np.int64),
          'kept_indices': keep.detach().cpu().numpy().astype(np.int64),
          'kept_mask': kept_mask.detach().cpu().numpy(),
          'angle_pass_mask': (alignment>=threshold).detach().cpu().numpy(),
          'threshold_deg': float(max_angle_deg),
      }
    poses_filtered = poses[keep]
    return poses_filtered, {
      'axis_prior_candidates_before': int(len(poses)),
      'axis_prior_candidates_after': int(len(poses_filtered)),
      'axis_prior_max_candidates': int(max_candidates),
      'axis_prior_max_alignment': float(alignment.max().detach().cpu()),
      'axis_prior_min_kept_alignment': float(alignment[keep].min().detach().cpu()),
      'axis_prior_max_angle_deg': float(max_angle_deg),
      'axis_prior_model_axis': [float(v) for v in model_axis],
    }


  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5, init_strategy='default', coarse_refine_iter=1, coarse_score_filter='none', coarse_score_top_k=999999, fine_refine_iter=2, fine_top_k=16, axis_prior_filter='none', axis_prior_model_axis=(0,0,1), axis_prior_max_angle_deg=45, axis_prior_min_candidates=12, axis_prior_max_candidates=0, axis_prior_min_points=500, axis_prior_min_confidence=1.4, axis_prior_debug=False):
    '''Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    '''
    timing = {}
    self.last_axis_prior_diagnostics = None
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
    valid = (depth>=0.001) & (ob_mask>0)
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
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
    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')
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
    if axis_prior_filter == 'depth_pca':
      t0 = time.perf_counter()
      scene_axis, axis_info = self.estimate_axis_prior_from_depth_pca(
          xyz_map=xyz_map,
          mask=ob_mask,
          min_points=axis_prior_min_points,
          min_confidence=axis_prior_min_confidence,
      )
      poses, filter_info = self.filter_pose_candidates_by_axis_prior(
          poses=poses,
          scene_axis=scene_axis,
          model_axis=axis_prior_model_axis,
          max_angle_deg=axis_prior_max_angle_deg,
          min_candidates=axis_prior_min_candidates,
          max_candidates=axis_prior_max_candidates,
          collect_diagnostics=axis_prior_debug,
      )
      torch.cuda.synchronize()
      timing['axis_prior'] = time.perf_counter() - t0
      timing.update(axis_info)
      timing.update(filter_info)
      timing['pose_hypothesis_candidates_after_axis_prior'] = len(poses)
    else:
      timing['axis_prior_status'] = 'disabled'
    if init_strategy == 'topk_two_stage':
      timing['refiner_coarse_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=coarse_refine_iter, get_vis=False)
      torch.cuda.synchronize()
      if self.last_axis_prior_diagnostics is not None:
        self.last_axis_prior_diagnostics['poses_after_coarse_refiner'] = poses.detach().cpu().numpy().astype(np.float32)
      timing['refiner_coarse'] = time.perf_counter() - t0
      timing['refiner_coarse_detail'] = dict(getattr(self.refiner, 'last_timing', {}))

      t0 = time.perf_counter()
      score_k = max(1, min(int(coarse_score_top_k), len(poses)))
      if coarse_score_filter == 'geometry' and score_k < len(poses):
        score_ids = self.select_coarse_score_candidates_by_geometry(poses, K, depth, ob_mask, score_k)
      else:
        score_ids = torch.linspace(0, len(poses) - 1, steps=score_k, device=poses.device).long()
      score_poses = poses[score_ids]
      if self.last_axis_prior_diagnostics is not None:
        self.last_axis_prior_diagnostics['coarse_selected_positions'] = score_ids.detach().cpu().numpy().astype(np.int64)
      torch.cuda.synchronize()
      timing['coarse_score_select'] = time.perf_counter() - t0
      timing['coarse_score_candidates'] = score_k
      timing['scorer_coarse_candidates'] = int(len(score_poses))
      timing['coarse_score_filter'] = coarse_score_filter

      t0 = time.perf_counter()
      scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=score_poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=False)
      torch.cuda.synchronize()
      timing['scorer_coarse_detail'] = dict(getattr(self.scorer, 'last_timing', {}))
      if self.last_axis_prior_diagnostics is not None:
        coarse_scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        self.last_axis_prior_diagnostics['coarse_scores'] = np.asarray(coarse_scores, dtype=np.float32).reshape(-1)
      timing['scorer_coarse'] = time.perf_counter() - t0

      t0 = time.perf_counter()
      top_k = max(1, min(int(fine_top_k), len(poses)))
      top_k = min(top_k, len(score_poses))
      top_ids = torch.as_tensor(scores).argsort(descending=True)[:top_k]
      poses = score_poses[top_ids]
      torch.cuda.synchronize()
      timing['topk_select'] = time.perf_counter() - t0

      timing['refiner_fine_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=fine_refine_iter, get_vis=self.debug>=2)
      torch.cuda.synchronize()
      timing['refiner_fine'] = time.perf_counter() - t0
      timing['refiner_fine_detail'] = dict(getattr(self.refiner, 'last_timing', {}))
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

      timing['scorer_fine_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
      torch.cuda.synchronize()
      timing['scorer_fine'] = time.perf_counter() - t0
      timing['scorer_fine_detail'] = dict(getattr(self.scorer, 'last_timing', {}))
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

      timing['refiner'] = timing['refiner_coarse'] + timing['refiner_fine']
      timing['scorer'] = timing['scorer_coarse'] + timing['scorer_fine']
      scorer_detail_keys = set(timing['scorer_coarse_detail']) | set(timing['scorer_fine_detail'])
      timing['scorer_detail'] = {
          key: timing['scorer_coarse_detail'].get(key, 0.0) + timing['scorer_fine_detail'].get(key, 0.0)
          for key in scorer_detail_keys
      }
      detail_keys = set(timing['refiner_coarse_detail']) | set(timing['refiner_fine_detail'])
      timing['refiner_detail'] = {
          key: timing['refiner_coarse_detail'].get(key, 0.0) + timing['refiner_fine_detail'].get(key, 0.0)
          for key in detail_keys
      }
    else:
      timing['refiner_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
      torch.cuda.synchronize()
      timing['refiner'] = time.perf_counter() - t0
      timing['refiner_detail'] = getattr(self.refiner, 'last_timing', {})
      if vis is not None:
        imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

      timing['scorer_candidates'] = int(len(poses))
      t0 = time.perf_counter()
      scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
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
    torch.cuda.synchronize()
    timing['sort_select'] = time.perf_counter() - t0

    torch.cuda.synchronize()
    timing['register'] = time.perf_counter() - t_register_start
    known_time = sum(timing.get(key, 0.0) for key in ('depth_preprocess', 'pose_hypothesis', 'axis_prior', 'refiner', 'coarse_score_select', 'scorer', 'topk_select', 'sort_select'))
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

    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    torch.cuda.synchronize()
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    self.last_track_timing = {'track': time.perf_counter() - t_track_start}
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)


