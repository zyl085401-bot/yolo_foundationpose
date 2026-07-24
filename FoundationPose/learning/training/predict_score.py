# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import functools
import os,sys,kornia
import time
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../')
from learning.datasets.h5_dataset import *
from learning.models.score_network import *
from learning.training.network_input_capture import NetworkInputCapture
try:
  from learning.training.tensorrt_runner import TensorRTEngineRunner
  _TENSORRT_IMPORT_ERROR = None
except Exception as error:
  TensorRTEngineRunner = None
  _TENSORRT_IMPORT_ERROR = error
from learning.datasets.pose_dataset import *
from Utils import *
from datareader import *


def vis_batch_data_scores(pose_data, ids, scores, pad_margin=5):
  assert len(scores)==len(ids)
  canvas = []
  for id in ids:
    rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
    rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
    H,W = rgbA_vis.shape[:2]
    zmin = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).min()
    zmax = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W).max()
    depthA_vis = depth_to_vis(pose_data.depthAs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    depthB_vis = depth_to_vis(pose_data.depthBs[id].data.cpu().numpy().reshape(H,W), zmin=zmin, zmax=zmax, inverse=False)
    if pose_data.normalAs is not None:
      pass
    pad = np.ones((rgbA_vis.shape[0],pad_margin,3))*255
    if pose_data.normalAs is not None:
      pass
    else:
      row = np.concatenate([rgbA_vis, pad, depthA_vis, pad, rgbB_vis, pad, depthB_vis], axis=1)
    s = 100/row.shape[0]
    row = cv2.resize(row, fx=s, fy=s, dsize=None)
    row = cv_draw_text(row, text=f'id:{id}, score:{scores[id]:.3f}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
    canvas.append(row)
    pad = np.ones((pad_margin, row.shape[1], 3))*255
    canvas.append(pad)
  canvas = np.concatenate(canvas, axis=0).astype(np.uint8)
  return canvas



@torch.no_grad()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, normal_map=None, xyz_map=None, mesh_diameter=None, glctx=None, mesh_tensors=None, dataset:TripletH5Dataset=None, cfg=None, timing=None, need_depthBs=True):
  logging.info("Welcome make_crop_data_batch")
  H,W = depth.shape[:2]

  args = []
  method = 'box_3d'
  t0 = time.perf_counter()
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_size[1], render_size[0]), method=method, mesh_diameter=mesh_diameter)
  if timing is not None:
    torch.cuda.synchronize()
    timing['crop_window'] += time.perf_counter() - t0
  logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  poseAs = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, cfg['input_resize'][0]-1, cfg['input_resize'][1]-1]).reshape(2,2), device='cuda', dtype=torch.float)
  bbox2d_ori = transform_pts(bbox2d_crop, tf_to_crops.inverse()[:,None]).reshape(-1,4)
  render_timing = timing if timing is not None and bool(cfg.get('render_profile_enabled', False)) else None

  t0 = time.perf_counter()
  for b in range(0,len(ob_in_cams),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseAs[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=cfg['input_resize'], bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra, render_timing=render_timing, batched_matmul_enabled=bool(cfg.get('render_batched_matmul_enabled', False)))
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    xyz_map_rs.append(extra['xyz_map'])
  if timing is not None:
    torch.cuda.synchronize()
    if render_timing is not None:
      finalize_nvdiffrast_render_timing(render_timing)
    timing['render'] += time.perf_counter() - t0

  t0 = time.perf_counter()
  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  if timing is not None:
    torch.cuda.synchronize()
    timing['render_postprocess'] += time.perf_counter() - t0
  logging.info("render done")

  t0 = time.perf_counter()
  rgbB_source = torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1)
  depthB_source = torch.as_tensor(depth, dtype=torch.float, device='cuda')[None,None].expand(B,-1,-1,-1)
  shared_warp_grid = None
  if bool(cfg.get('scorer_shared_warp_grid_enabled', False)):
    shared_warp_grid = build_warp_perspective_grid(rgbB_source, tf_to_crops, render_size)
    rgbBs = warp_perspective_from_grid(rgbB_source, shared_warp_grid, mode='bilinear')
    depthBs = warp_perspective_from_grid(depthB_source, shared_warp_grid, mode='nearest') if need_depthBs else None
  else:
    rgbBs = kornia.geometry.transform.warp_perspective(rgbB_source, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthBs = kornia.geometry.transform.warp_perspective(depthB_source, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False) if need_depthBs else None
  if rgb_rs.shape[-2:]!=cfg['input_resize']:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
    depthAs = kornia.geometry.transform.warp_perspective(depth_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    rgbAs = rgb_rs
    depthAs = depth_rs

  if xyz_map_rs.shape[-2:]!=cfg['input_resize']:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs

  normalAs = None
  normalBs = None
  if timing is not None:
    torch.cuda.synchronize()
    timing['warp'] += time.perf_counter() - t0

  t0 = time.perf_counter()
  transform_profile_events = []
  if timing is not None and bool(cfg.get('render_profile_enabled', False)):
    def record_transform_event(name):
      event = torch.cuda.Event(enable_timing=True)
      event.record()
      transform_profile_events.append((name, event))

    record_transform_event(None)
  else:
    def record_transform_event(name):
      return None

  xyz_mapBs = None
  if xyz_map is not None and bool(cfg.get('scorer_precomputed_xyz_enabled', False)):
    xyz_map_full = torch.as_tensor(xyz_map, dtype=torch.float, device='cuda').permute(2,0,1)[None]
    xyz_mapB_source = xyz_map_full.expand(B,-1,-1,-1)
    if shared_warp_grid is not None:
      xyz_mapBs = warp_perspective_from_grid(xyz_mapB_source, shared_warp_grid, mode='nearest')
    else:
      xyz_mapBs = kornia.geometry.transform.warp_perspective(
          xyz_mapB_source,
          tf_to_crops,
          dsize=render_size,
          mode='nearest',
          align_corners=False,
      )
    record_transform_event('transform_xyzB_precomputed_warp_crop')

  Ks = torch.as_tensor(K, dtype=torch.float).reshape(1,3,3).expand(len(rgbAs),3,3)
  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter

  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=depthAs, depthBs=depthBs, normalAs=normalAs, normalBs=normalBs, poseA=poseAs, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  record_transform_event('transform_batch_setup')
  pose_data = dataset.transform_batch(
      pose_data,
      H_ori=H,
      W_ori=W,
      bound=1,
      transform_event=record_transform_event if transform_profile_events else None,
  )
  if timing is not None:
    torch.cuda.synchronize()
    timing['transform'] += time.perf_counter() - t0
    if transform_profile_events:
      previous_event = transform_profile_events[0][1]
      transform_profile_keys = []
      for name, event in transform_profile_events[1:]:
        timing[name] = timing.get(name, 0.0) + previous_event.elapsed_time(event) / 1000.0
        transform_profile_keys.append(name)
        previous_event = event
      timing['transform_profile_other'] = max(
          timing['transform'] - sum(timing.get(key, 0.0) for key in transform_profile_keys),
          0.0,
      )

  logging.info("pose batch data done")

  return pose_data


class ScorePredictor:
  def __init__(self, amp=True, network_input_capture=None, tensorrt_backend=None, render_profile_enabled=False, render_batched_matmul_enabled=False, scorer_precomputed_xyz_enabled=False, scorer_shared_warp_grid_enabled=False, scorer_skip_unused_depth_warp_enabled=False):
    self.amp = amp
    self.run_name = "2024-01-11-20-02-45"

    model_name = 'model_best.pth'
    code_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_dir = f'{code_dir}/../../weights/{self.run_name}/{model_name}'

    self.cfg = OmegaConf.load(f'{code_dir}/../../weights/{self.run_name}/config.yml')

    self.cfg['ckpt_dir'] = ckpt_dir
    self.cfg['enable_amp'] = True
    self.render_profile_enabled = bool(render_profile_enabled)
    self.cfg['render_profile_enabled'] = self.render_profile_enabled
    self.render_batched_matmul_enabled = bool(render_batched_matmul_enabled)
    self.cfg['render_batched_matmul_enabled'] = self.render_batched_matmul_enabled
    self.scorer_precomputed_xyz_enabled = bool(scorer_precomputed_xyz_enabled)
    self.cfg['scorer_precomputed_xyz_enabled'] = self.scorer_precomputed_xyz_enabled
    self.scorer_shared_warp_grid_enabled = bool(scorer_shared_warp_grid_enabled)
    self.cfg['scorer_shared_warp_grid_enabled'] = self.scorer_shared_warp_grid_enabled
    self.scorer_skip_unused_depth_warp_enabled = bool(scorer_skip_unused_depth_warp_enabled)
    self.cfg['scorer_skip_unused_depth_warp_enabled'] = self.scorer_skip_unused_depth_warp_enabled

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = np.inf
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 4
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2

    logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = ScoreMultiPairH5Dataset(cfg=self.cfg, mode='test', h5_file=None, max_num_key=1)
    self.model = ScoreNetMultiPair(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()

    logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)

    self.model.cuda().eval()
    self.model.fuse_conv_batchnorm()
    self.model.to(memory_format=torch.channels_last)
    self.last_timing = {}
    self.network_input_capture = NetworkInputCapture('scorer', network_input_capture)
    self._configure_tensorrt(tensorrt_backend, ckpt_dir)
    logging.info("init done")


  def _configure_tensorrt(self, config, checkpoint_path):
    config = dict(config or {})
    enabled_override = os.getenv('FOUNDATIONPOSE_SCORER_TENSORRT')
    self.tensorrt_enabled = bool(config.get('enabled', False)) if enabled_override is None else enabled_override.strip().lower() in ('1', 'true', 'yes', 'on')
    self.tensorrt_runners = {}
    self.tensorrt_failure_reasons = {}
    if not self.tensorrt_enabled:
      return
    if TensorRTEngineRunner is None:
      reason = f'TensorRT import failed: {_TENSORRT_IMPORT_ERROR}'
      self.tensorrt_failure_reasons['all'] = reason
      logging.warning(f'Scorer {reason}; using PyTorch')
      return
    for candidate_count_value, engine_path in dict(config.get('engine_paths', {})).items():
      candidate_count = int(candidate_count_value)
      try:
        self.tensorrt_runners[candidate_count] = TensorRTEngineRunner(
            engine_path,
            expected_network='scorer',
            expected_checkpoint_path=checkpoint_path,
        )
        logging.info(f'Scorer N={candidate_count} TensorRT backend loaded: {self.tensorrt_runners[candidate_count].engine_path}')
      except Exception as error:
        reason = f'TensorRT initialization failed for N={candidate_count}: {type(error).__name__}: {error}'
        self.tensorrt_failure_reasons[candidate_count] = reason
        logging.warning(f'Scorer {reason}; using PyTorch for this shape')


  def _run_network(self, A, B):
    candidate_count = int(A.shape[0])
    detail = {
        'layout_convert': 0.0,
        'dtype_convert': 0.0,
        'tensorrt_execute': 0.0,
    }
    fallback_reason = None
    runner = self.tensorrt_runners.get(candidate_count)
    if runner is not None:
      try:
        dtype_start = torch.cuda.Event(enable_timing=True)
        dtype_end = torch.cuda.Event(enable_timing=True)
        layout_start = torch.cuda.Event(enable_timing=True)
        layout_end = torch.cuda.Event(enable_timing=True)
        dtype_start.record()
        A_trt = A.float()
        B_trt = B.float()
        dtype_end.record()
        layout_start.record()
        A_trt = A_trt.contiguous()
        B_trt = B_trt.contiguous()
        layout_end.record()
        output = runner({'A': A_trt, 'B': B_trt})
        torch.cuda.synchronize()
        detail['dtype_convert'] = dtype_start.elapsed_time(dtype_end) / 1000.0
        detail['layout_convert'] = layout_start.elapsed_time(layout_end) / 1000.0
        detail['tensorrt_execute'] = runner.collect_last_cuda_timing() or 0.0
        return output, 'tensorrt', None, detail
      except Exception as error:
        fallback_reason = f'TensorRT runtime failed for N={candidate_count}: {type(error).__name__}: {error}'
        self.tensorrt_failure_reasons[candidate_count] = fallback_reason
        self.tensorrt_runners.pop(candidate_count, None)
        logging.exception(f'Scorer {fallback_reason}; disabling this engine and using PyTorch')
    elif self.tensorrt_enabled:
      fallback_reason = self.tensorrt_failure_reasons.get(candidate_count)
      fallback_reason = fallback_reason or self.tensorrt_failure_reasons.get('all')
      fallback_reason = fallback_reason or f'unsupported candidate count N={candidate_count}; fixed engines are required'

    with torch.cuda.amp.autocast(enabled=self.amp):
      output = self.model(A, B, L=candidate_count)
    torch.cuda.synchronize()
    detail.update(self.model.collect_last_cuda_timing())
    return output, 'pytorch', fallback_reason, detail


  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, normal_map=None, xyz_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, network_stage='scorer'):
    '''
    @rgb: np array (H,W,3)
    '''
    logging.info(f"ob_in_cams:{ob_in_cams.shape}")
    timing = {
      'crop_window': 0.0,
      'render': 0.0,
      'render_postprocess': 0.0,
      'warp': 0.0,
      'transform': 0.0,
      'input_pack': 0.0,
      'network_forward': 0.0,
      'encoderA': 0.0,
      'encoderAB': 0.0,
      'self_attention': 0.0,
      'cross_attention': 0.0,
      'linear': 0.0,
      'layout_convert': 0.0,
      'dtype_convert': 0.0,
      'tensorrt_execute': 0.0,
      'output_convert': 0.0,
      'empty_cache': 0.0,
      'total': 0.0,
      'other': 0.0,
    }
    network_backends = set()
    fallback_reasons = set()
    torch.cuda.synchronize()
    total_start = time.perf_counter()
    ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

    logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    logging.info("making cropped data")

    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh)

    rgb = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)

    precomputed_xyz_enabled = xyz_map is not None and bool(self.cfg.get('scorer_precomputed_xyz_enabled', False))
    skip_unused_depth_warp = bool(self.cfg.get('scorer_skip_unused_depth_warp_enabled', False)) and precomputed_xyz_enabled and not get_vis
    pose_data = make_crop_data_batch(self.cfg.input_resize, ob_in_cams, mesh, rgb, depth, K, crop_ratio=self.cfg['crop_ratio'], glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, cfg=self.cfg, mesh_diameter=mesh_diameter, timing=timing, xyz_map=xyz_map, need_depthBs=not skip_unused_depth_warp)

    def find_best_among_pairs(pose_data:BatchPoseData):
      logging.info(f'pose_data.rgbAs.shape[0]: {pose_data.rgbAs.shape[0]}')
      ids = []
      scores = []
      bs = pose_data.rgbAs.shape[0]
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        t0 = time.perf_counter()
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float().contiguous(memory_format=torch.channels_last)
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float().contiguous(memory_format=torch.channels_last)
        if pose_data.normalAs is not None:
          A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1).contiguous(memory_format=torch.channels_last)
          B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1).contiguous(memory_format=torch.channels_last)
        torch.cuda.synchronize()
        timing['input_pack'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        output, network_backend, fallback_reason, network_detail = self._run_network(A, B)
        timing['network_forward'] += time.perf_counter() - t0
        network_backends.add(network_backend)
        if fallback_reason:
          fallback_reasons.add(fallback_reason)
        for name, elapsed in network_detail.items():
          timing[name] += elapsed
        output_start = torch.cuda.Event(enable_timing=True)
        output_end = torch.cuda.Event(enable_timing=True)
        output_start.record()
        scores_cur = output["score_logit"].float().reshape(-1)
        output_end.record()
        torch.cuda.synchronize()
        timing['output_convert'] += output_start.elapsed_time(output_end) / 1000.0
        self.network_input_capture.capture(
            stage=network_stage,
            A=A,
            B=B,
            output=output,
            L=len(A),
        )
        ids.append(scores_cur.argmax()+b)
        scores.append(scores_cur)
      ids = torch.stack(ids, dim=0).reshape(-1)
      scores = torch.cat(scores, dim=0).reshape(-1)
      return ids, scores

    pose_data_iter = pose_data
    global_ids = torch.arange(len(ob_in_cams), device='cuda', dtype=torch.long)
    scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device='cuda')

    while 1:
      ids, scores = find_best_among_pairs(pose_data_iter)
      if len(ids)==1:
        scores_global[global_ids] = scores + 100
        break
      global_ids = global_ids[ids]
      pose_data_iter = pose_data.select_by_indices(global_ids)

    scores = scores_global

    logging.info(f'forward done')

    def finalize_timing():
      torch.cuda.synchronize()
      timing['total'] = time.perf_counter() - total_start
      if self.render_profile_enabled:
        render_profile_keys = (
            'render_context_mesh_check',
            'render_projection_setup',
          'render_vertex_camera_transform',
          'render_vertex_homogeneous',
          'render_vertex_clip_transform',
            'render_bbox_transform',
            'render_rasterize',
            'render_xyz_depth_interpolate',
            'render_texture_sample',
          'render_normal_transform',
          'render_normal_interpolate',
          'render_normal_normalize_flip',
          'render_diffuse_vertex',
          'render_diffuse_interpolate',
          'render_lighting_blend',
            'render_finalize_flip_mask',
        )
        timing['render_profile_other'] = max(
            timing['render'] - sum(timing.get(key, 0.0) for key in render_profile_keys),
            0.0,
        )
      top_level_keys = (
        'crop_window', 'render', 'render_postprocess', 'warp', 'transform',
        'input_pack', 'network_forward', 'output_convert', 'empty_cache',
      )
      timing['other'] = max(0.0, timing['total'] - sum(timing[key] for key in top_level_keys))
      timing['network_backend'] = next(iter(network_backends)) if len(network_backends) == 1 else 'mixed'
      timing['fallback_reason'] = '; '.join(sorted(fallback_reasons)) or None
      if network_backends == {'tensorrt'}:
        for name in ('encoderA', 'encoderAB', 'self_attention', 'cross_attention', 'linear'):
          timing[name] = None
      elif network_backends == {'pytorch'}:
        for name in ('layout_convert', 'dtype_convert', 'tensorrt_execute'):
          timing[name] = None
      self.last_timing = dict(timing)

    if get_vis:
      logging.info("get_vis...")
      canvas = []
      ids = scores.argsort(descending=True)
      canvas = vis_batch_data_scores(pose_data, ids=ids, scores=scores)
      finalize_timing()
      return scores, canvas

    finalize_timing()
    return scores, None

