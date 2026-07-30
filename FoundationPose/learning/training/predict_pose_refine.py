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
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
import numpy as np
import torch
from omegaconf import OmegaConf
from learning.models.refine_network import RefineNet
from learning.training.network_input_capture import NetworkInputCapture
try:
  from learning.training.tensorrt_runner import TensorRTEngineRunner
  _TENSORRT_IMPORT_ERROR = None
except Exception as error:
  TensorRTEngineRunner = None
  _TENSORRT_IMPORT_ERROR = error
from learning.datasets.h5_dataset import *
from Utils import *
from datareader import *


def start_cuda_stage_timing(timing):
  if timing is None:
    return None
  if timing.get('_cuda_stage_events_enabled', False):
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return 'cuda', event
  return 'wall', time.perf_counter()


def finish_cuda_stage_timing(timing, name, start_event):
  if timing is None or start_event is None:
    return
  mode, start_value = start_event
  if mode == 'wall':
    torch.cuda.synchronize()
    timing[name] += time.perf_counter() - start_value
    return
  end_event = torch.cuda.Event(enable_timing=True)
  end_event.record()
  timing.setdefault('_cuda_stage_events', []).append((name, start_value, end_event))


def finalize_cuda_stage_timing(timing):
  for name, start_event, end_event in timing.pop('_cuda_stage_events', []):
    timing[name] += start_event.elapsed_time(end_event) / 1000.0



@torch.inference_mode()
def make_crop_data_batch(render_size, ob_in_cams, mesh, rgb, depth, K, crop_ratio, xyz_map, normal_map=None, mesh_diameter=None, cfg=None, glctx=None, mesh_tensors=None, dataset:PoseRefinePairH5Dataset=None, timing=None):
  logging.info("Welcome make_crop_data_batch")
  render_size = tuple(int(value) for value in render_size)
  render_height, render_width = render_size
  H,W = depth.shape[:2]
  args = []
  method = 'box_3d'
  stage_start = start_cuda_stage_timing(timing)
  tf_to_crops = compute_crop_window_tf_batch(pts=mesh.vertices, H=H, W=W, poses=ob_in_cams, K=K, crop_ratio=crop_ratio, out_size=(render_width, render_height), method=method, mesh_diameter=mesh_diameter)
  finish_cuda_stage_timing(timing, 'crop_window', stage_start)

  logging.info("make tf_to_crops done")

  B = len(ob_in_cams)
  poseA = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

  bs = 512
  rgb_rs = []
  depth_rs = []
  normal_rs = []
  xyz_map_rs = []

  bbox2d_crop = torch.as_tensor(np.array([0, 0, render_width-1, render_height-1]).reshape(2,2), device='cuda', dtype=torch.float)
  crop_to_oris = tf_to_crops.inverse()
  bbox2d_ori = transform_pts(bbox2d_crop, crop_to_oris).reshape(-1,4)
  render_timing = timing if timing is not None and bool(cfg.get('render_profile_enabled', False)) else None

  stage_start = start_cuda_stage_timing(timing)
  for b in range(0,len(poseA),bs):
    extra = {}
    rgb_r, depth_r, normal_r = nvdiffrast_render(K=K, H=H, W=W, ob_in_cams=poseA[b:b+bs], context='cuda', get_normal=cfg['use_normal'], glctx=glctx, mesh_tensors=mesh_tensors, output_size=render_size, bbox2d=bbox2d_ori[b:b+bs], use_light=True, extra=extra, render_timing=render_timing, batched_matmul_enabled=bool(cfg.get('render_batched_matmul_enabled', False)))
    rgb_rs.append(rgb_r)
    depth_rs.append(depth_r[...,None])
    normal_rs.append(normal_r)
    xyz_map_rs.append(extra['xyz_map'])
  finish_cuda_stage_timing(timing, 'render', stage_start)

  stage_start = start_cuda_stage_timing(timing)
  rgb_rs = torch.cat(rgb_rs, dim=0).permute(0,3,1,2) * 255
  depth_rs = torch.cat(depth_rs, dim=0).permute(0,3,1,2)  #(B,1,H,W)
  xyz_map_rs = torch.cat(xyz_map_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  Ks = torch.as_tensor(K, device='cuda', dtype=torch.float).reshape(1,3,3)
  if cfg['use_normal']:
    normal_rs = torch.cat(normal_rs, dim=0).permute(0,3,1,2)  #(B,3,H,W)
  finish_cuda_stage_timing(timing, 'render_postprocess', stage_start)

  logging.info("render done")

  stage_start = start_cuda_stage_timing(timing)
  rgbB_source = torch.as_tensor(rgb, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1)
  xyz_mapB_source = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float).permute(2,0,1)[None].expand(B,-1,-1,-1)
  shared_warp_grid = None
  if bool(cfg.get('refiner_shared_warp_grid_enabled', False)):
    shared_warp_grid = build_warp_perspective_grid(rgbB_source, tf_to_crops, render_size)
    rgbBs = warp_perspective_from_grid(rgbB_source, shared_warp_grid, mode='bilinear')
  else:
    rgbBs = kornia.geometry.transform.warp_perspective(rgbB_source, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  if tuple(rgb_rs.shape[-2:])!=render_size:
    rgbAs = kornia.geometry.transform.warp_perspective(rgb_rs, tf_to_crops, dsize=render_size, mode='bilinear', align_corners=False)
  else:
    rgbAs = rgb_rs
  if tuple(xyz_map_rs.shape[-2:])!=render_size:
    xyz_mapAs = kornia.geometry.transform.warp_perspective(xyz_map_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    xyz_mapAs = xyz_map_rs
  if shared_warp_grid is not None:
    xyz_mapBs = warp_perspective_from_grid(xyz_mapB_source, shared_warp_grid, mode='nearest')
  else:
    xyz_mapBs = kornia.geometry.transform.warp_perspective(xyz_mapB_source, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)  #(B,3,H,W)

  if cfg['use_normal']:
    normalAs = kornia.geometry.transform.warp_perspective(normal_rs, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
    normalB_source = torch.as_tensor(normal_map, dtype=torch.float, device='cuda').permute(2,0,1)[None].expand(B,-1,-1,-1)
    if shared_warp_grid is not None:
      normalBs = warp_perspective_from_grid(normalB_source, shared_warp_grid, mode='nearest')
    else:
      normalBs = kornia.geometry.transform.warp_perspective(normalB_source, tf_to_crops, dsize=render_size, mode='nearest', align_corners=False)
  else:
    normalAs = None
    normalBs = None
  finish_cuda_stage_timing(timing, 'warp', stage_start)

  logging.info("warp done")

  stage_start = start_cuda_stage_timing(timing)
  mesh_diameters = torch.ones((len(rgbAs)), dtype=torch.float, device='cuda')*mesh_diameter
  pose_data = BatchPoseData(rgbAs=rgbAs, rgbBs=rgbBs, depthAs=None, depthBs=None, normalAs=normalAs, normalBs=normalBs, poseA=poseA, poseB=None, xyz_mapAs=xyz_mapAs, xyz_mapBs=xyz_mapBs, tf_to_crops=tf_to_crops, Ks=Ks, mesh_diameters=mesh_diameters)
  pose_data.crop_to_oris = crop_to_oris
  pose_data.Ks_inv = Ks.inverse()
  pose_data = dataset.transform_batch(batch=pose_data, H_ori=H, W_ori=W, bound=1)
  finish_cuda_stage_timing(timing, 'transform', stage_start)

  logging.info("pose batch data done")

  return pose_data



class PoseRefinePredictor:
  def __init__(self, network_input_capture=None, tensorrt_backend=None, input_sizes=None, render_profile_enabled=False, render_batched_matmul_enabled=False, refiner_stage1_optimizations_enabled=False, refiner_shared_warp_grid_enabled=False, network_internal_sync_enabled=True):
    logging.info("welcome")
    self.amp = True
    self.run_name = "2023-10-28-18-33-37"
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
    self.refiner_stage1_optimizations_enabled = bool(refiner_stage1_optimizations_enabled)
    self.cfg['refiner_stage1_optimizations_enabled'] = self.refiner_stage1_optimizations_enabled
    self.refiner_shared_warp_grid_enabled = bool(refiner_shared_warp_grid_enabled)
    self.cfg['refiner_shared_warp_grid_enabled'] = self.refiner_shared_warp_grid_enabled
    self.network_internal_sync_enabled = bool(network_internal_sync_enabled)

    ########## Defaults, to be backward compatible
    if 'use_normal' not in self.cfg:
      self.cfg['use_normal'] = False
    if 'use_mask' not in self.cfg:
      self.cfg['use_mask'] = False
    if 'use_BN' not in self.cfg:
      self.cfg['use_BN'] = False
    if 'c_in' not in self.cfg:
      self.cfg['c_in'] = 4
    if 'crop_ratio' not in self.cfg or self.cfg['crop_ratio'] is None:
      self.cfg['crop_ratio'] = 1.2
    if 'n_view' not in self.cfg:
      self.cfg['n_view'] = 1
    if 'trans_rep' not in self.cfg:
      self.cfg['trans_rep'] = 'tracknet'
    if 'rot_rep' not in self.cfg:
      self.cfg['rot_rep'] = 'axis_angle'
    if 'zfar' not in self.cfg:
      self.cfg['zfar'] = 3
    if 'normalize_xyz' not in self.cfg:
      self.cfg['normalize_xyz'] = False
    if isinstance(self.cfg['zfar'], str) and 'inf' in self.cfg['zfar'].lower():
      self.cfg['zfar'] = np.inf
    if 'normal_uint8' not in self.cfg:
      self.cfg['normal_uint8'] = False
    self.default_input_size = self._normalize_input_size(self.cfg['input_resize'], 'default')
    self.input_sizes = {
        str(stage): self._normalize_input_size(input_size, str(stage))
        for stage, input_size in dict(input_sizes or {}).items()
    }
    logging.info(f'Refiner input sizes: default={self.default_input_size}, stages={self.input_sizes}')
    logging.info(f"self.cfg: \n {OmegaConf.to_yaml(self.cfg)}")

    self.dataset = PoseRefinePairH5Dataset(cfg=self.cfg, h5_file='', mode='test')
    self.model = RefineNet(cfg=self.cfg, c_in=self.cfg['c_in']).cuda()

    logging.info(f"Using pretrained model from {ckpt_dir}")
    ckpt = torch.load(ckpt_dir)
    if 'model' in ckpt:
      ckpt = ckpt['model']
    self.model.load_state_dict(ckpt)

    self.model.cuda().eval()
    self.model.fuse_conv_batchnorm()
    self.model.to(memory_format=torch.channels_last)
    logging.info("init done")
    self.last_trans_update = None
    self.last_rot_update = None
    self.last_timing = {}
    self.network_input_capture = NetworkInputCapture('refiner', network_input_capture)
    self._configure_tensorrt(tensorrt_backend, ckpt_dir)


  @staticmethod
  def _normalize_input_size(input_size, stage):
    try:
      normalized = tuple(int(value) for value in input_size)
    except (TypeError, ValueError) as error:
      raise ValueError(f'Refiner input size for {stage} must contain height and width, got {input_size!r}') from error
    if len(normalized) != 2 or any(value <= 0 for value in normalized):
      raise ValueError(f'Refiner input size for {stage} must be two positive integers, got {input_size!r}')
    height, width = normalized
    token_count = ((height + 7) // 8) * ((width + 7) // 8)
    if token_count > 400:
      raise ValueError(f'Refiner input size for {stage} produces {token_count} tokens, exceeding positional embedding limit 400')
    return normalized


  def _input_size_for_stage(self, network_stage):
    return self.input_sizes.get(network_stage, self.default_input_size)


  def _validate_tensorrt_input_size(self, runner, network_stage):
    input_shape = runner.metadata.get('input_shape')
    if not isinstance(input_shape, list) or len(input_shape) < 4:
      return
    engine_input_size = tuple(int(value) for value in input_shape[-2:])
    expected_input_size = self._input_size_for_stage(network_stage)
    if engine_input_size != expected_input_size:
      raise RuntimeError(
          f'engine input size {engine_input_size} does not match {network_stage} input size {expected_input_size}'
      )


  def _configure_tensorrt(self, config, checkpoint_path):
    config = dict(config or {})
    enabled_override = os.getenv('FOUNDATIONPOSE_REFINER_TENSORRT')
    self.tensorrt_enabled = bool(config.get('enabled', False)) if enabled_override is None else enabled_override.strip().lower() in ('1', 'true', 'yes', 'on')
    self.tensorrt_min_candidates = int(config.get('min_candidates', 5))
    self.tensorrt_max_candidates = int(config.get('max_candidates', 12))
    stages_override = os.getenv('FOUNDATIONPOSE_REFINER_TENSORRT_STAGES')
    configured_stages = config.get('stages', ('refiner_coarse', 'refiner_fine'))
    self.tensorrt_stages = set(configured_stages if stages_override is None else stages_override.split(','))
    self.tensorrt_stages = {stage.strip() for stage in self.tensorrt_stages if stage.strip()}
    self.tensorrt_runners = {}
    self.tensorrt_failure_reasons = {}
    if not self.tensorrt_enabled:
      return
    if TensorRTEngineRunner is None:
      reason = f'TensorRT import failed: {_TENSORRT_IMPORT_ERROR}'
      self.tensorrt_failure_reasons['all'] = reason
      logging.warning(f'Refiner {reason}; using PyTorch')
      return
    legacy_engine_path = config.get('engine_path')
    engine_paths = {
        str(stage): engine_path
        for stage, engine_path in dict(config.get('engine_paths', {})).items()
    }
    runners_by_path = {}
    for network_stage in sorted(self.tensorrt_stages):
      engine_path = engine_paths.get(network_stage, legacy_engine_path)
      if not engine_path:
        reason = f'no TensorRT engine configured for {network_stage}'
        self.tensorrt_failure_reasons[network_stage] = reason
        logging.warning(f'Refiner {reason}; using PyTorch for this stage')
        continue
      try:
        runner = runners_by_path.get(engine_path)
        if runner is None:
          runner = TensorRTEngineRunner(
              engine_path,
              expected_network='refiner',
              expected_checkpoint_path=checkpoint_path,
          )
          runners_by_path[engine_path] = runner
        self._validate_tensorrt_input_size(runner, network_stage)
        self.tensorrt_runners[network_stage] = runner
        logging.info(f'Refiner {network_stage} TensorRT backend loaded: {runner.engine_path}')
      except Exception as error:
        reason = f'TensorRT initialization failed: {type(error).__name__}: {error}'
        self.tensorrt_failure_reasons[network_stage] = reason
        logging.warning(f'Refiner {network_stage} {reason}; using PyTorch for this stage')


  def _uses_tensorrt(self, candidate_count, network_stage):
    return (
        self.tensorrt_runners.get(network_stage) is not None
        and self.tensorrt_min_candidates <= candidate_count <= self.tensorrt_max_candidates
    )


  def _run_network(self, A, B, network_stage):
    candidate_count = int(A.shape[0])
    detail = {
        'layout_convert': 0.0,
        'dtype_convert': 0.0,
        'tensorrt_execute': 0.0,
    }
    fallback_reason = None
    if self._uses_tensorrt(candidate_count, network_stage):
      runner = self.tensorrt_runners[network_stage]
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
        tensorrt_events = runner.pop_last_cuda_timing_events()
        if self.network_internal_sync_enabled:
          torch.cuda.synchronize()
          detail['dtype_convert'] = dtype_start.elapsed_time(dtype_end) / 1000.0
          detail['layout_convert'] = layout_start.elapsed_time(layout_end) / 1000.0
          detail['tensorrt_execute'] = (
              tensorrt_events[0].elapsed_time(tensorrt_events[1]) / 1000.0
              if tensorrt_events is not None else 0.0
          )
        else:
          detail['_deferred_cuda_timing_events'] = (
              ('dtype_convert', dtype_start, dtype_end),
              ('layout_convert', layout_start, layout_end),
          )
          if tensorrt_events is not None:
            detail['_deferred_cuda_timing_events'] += (
                ('tensorrt_execute', tensorrt_events[0], tensorrt_events[1]),
            )
        return output, 'tensorrt', None, detail
      except Exception as error:
        fallback_reason = f'TensorRT runtime failed: {type(error).__name__}: {error}'
        self.tensorrt_failure_reasons[network_stage] = fallback_reason
        self.tensorrt_runners.pop(network_stage, None)
        logging.exception(f'Refiner {network_stage} {fallback_reason}; disabling TensorRT for this stage and using PyTorch')
    elif self.tensorrt_enabled:
      if network_stage not in self.tensorrt_stages:
        fallback_reason = f'unsupported stage {network_stage}'
      else:
        fallback_reason = self.tensorrt_failure_reasons.get(network_stage)
        fallback_reason = fallback_reason or self.tensorrt_failure_reasons.get('all')
        fallback_reason = fallback_reason or f'unsupported candidate count N={candidate_count}'

    with torch.cuda.amp.autocast(enabled=self.amp):
      output = self.model(A, B)
    torch.cuda.synchronize()
    detail.update(self.model.collect_last_cuda_timing())
    return output, 'pytorch', fallback_reason, detail


  @torch.inference_mode()
  def predict(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None, get_vis=False, mesh=None, mesh_tensors=None, glctx=None, mesh_diameter=None, iteration=5, network_stage='refiner', capture_iteration_poses=False):
    '''
    @rgb: np array (H,W,3)
    @ob_in_cams: np array (N,4,4)
    '''
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    logging.info(f'ob_in_cams:{ob_in_cams.shape}')
    tf_to_center = np.eye(4)
    ob_centered_in_cams = ob_in_cams
    mesh_centered = mesh

    logging.info(f'self.cfg.use_normal:{self.cfg.use_normal}')
    if not self.cfg.use_normal:
      normal_map = None

    crop_ratio = self.cfg['crop_ratio']
    stage_input_size = self._input_size_for_stage(network_stage)
    logging.info(f'Refiner {network_stage} input size: {stage_input_size}')
    logging.info(f"trans_normalizer:{self.cfg['trans_normalizer']}, rot_normalizer:{self.cfg['rot_normalizer']}")
    bs = 1024

    B_in_cams = torch.as_tensor(ob_centered_in_cams, device='cuda', dtype=torch.float)
    self.last_iteration_poses = []


    if mesh_tensors is None:
      mesh_tensors = make_mesh_tensors(mesh_centered)

    rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
    depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)
    trans_normalizer = self.cfg['trans_normalizer']
    if not isinstance(trans_normalizer, float):
      trans_normalizer = torch.as_tensor(list(trans_normalizer), device='cuda', dtype=torch.float).reshape(1,3)

    timing = {
        'crop_window': 0.0,
        'render': 0.0,
        'render_postprocess': 0.0,
        'warp': 0.0,
        'transform': 0.0,
        'input_pack': 0.0,
        'network_forward': 0.0,
        'encodeA': 0.0,
        'encodeAB': 0.0,
        'trans_head': 0.0,
        'rot_head': 0.0,
        'layout_convert': 0.0,
        'dtype_convert': 0.0,
        'tensorrt_execute': 0.0,
        'output_convert': 0.0,
        'pose_update': 0.0,
        'empty_cache': 0.0,
        'total': 0.0,
        'other': 0.0,
        'network_internal_sync_status': 'enabled' if self.network_internal_sync_enabled else 'disabled',
        'network_stage': network_stage,
        'input_size': f'{stage_input_size[0]}x{stage_input_size[1]}',
        '_cuda_stage_events_enabled': self.refiner_stage1_optimizations_enabled,
    }
    network_backends = set()
    fallback_reasons = set()
    deferred_cuda_timing_events = []
    total_start = time.perf_counter()

    for iteration_index in range(iteration):
      logging.info("making cropped data")
      pose_data = make_crop_data_batch(stage_input_size, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter, timing=timing)
      B_in_cams = []
      for b in range(0, pose_data.rgbAs.shape[0], bs):
        stage_start = start_cuda_stage_timing(timing)
        candidate_count = len(pose_data.rgbAs[b:b+bs])
        input_memory_format = torch.contiguous_format if self.refiner_stage1_optimizations_enabled and self._uses_tensorrt(candidate_count, network_stage) else torch.channels_last
        A = torch.cat([pose_data.rgbAs[b:b+bs].cuda(), pose_data.xyz_mapAs[b:b+bs].cuda()], dim=1).float().contiguous(memory_format=input_memory_format)
        B = torch.cat([pose_data.rgbBs[b:b+bs].cuda(), pose_data.xyz_mapBs[b:b+bs].cuda()], dim=1).float().contiguous(memory_format=input_memory_format)
        finish_cuda_stage_timing(timing, 'input_pack', stage_start)

        logging.info("forward start")
        stage_start = start_cuda_stage_timing(timing)
        output, network_backend, fallback_reason, network_detail = self._run_network(A, B, network_stage)
        finish_cuda_stage_timing(timing, 'network_forward', stage_start)
        network_backends.add(network_backend)
        if fallback_reason:
          fallback_reasons.add(fallback_reason)
        deferred_cuda_timing_events.extend(
            network_detail.pop('_deferred_cuda_timing_events', ())
        )
        for name, elapsed in network_detail.items():
          if elapsed is not None:
            timing[name] += elapsed
        self.network_input_capture.capture(
            stage=f'{network_stage}_iter{iteration_index + 1}',
            A=A,
            B=B,
            output=output,
            iteration=iteration_index + 1,
        )
        stage_start = start_cuda_stage_timing(timing)
        for k in output:
          output[k] = output[k].float()
        finish_cuda_stage_timing(timing, 'output_convert', stage_start)
        logging.info("forward done")

        stage_start = start_cuda_stage_timing(timing)
        if self.cfg['trans_rep']=='tracknet':
          if not self.cfg['normalize_xyz']:
            trans_delta = torch.tanh(output["trans"])*trans_normalizer
          else:
            trans_delta = output["trans"]

        elif self.cfg['trans_rep']=='deepim':
          def project_and_transform_to_crop(centers):
            uvs = (pose_data.Ks[b:b+bs]@centers.reshape(-1,3,1)).reshape(-1,3)
            uvs = uvs/uvs[:,2:3]
            uvs = (pose_data.tf_to_crops[b:b+bs]@uvs.reshape(-1,3,1)).reshape(-1,3)
            return uvs[:,:2]

          rot_delta = output["rot"]
          z_pred = output['trans'][:,2]*pose_data.poseA[b:b+bs][...,2,3]
          uvA_crop = project_and_transform_to_crop(pose_data.poseA[b:b+bs][...,:3,3])
          uv_pred_crop = uvA_crop + output['trans'][:,:2]*stage_input_size[1]
          if self.refiner_stage1_optimizations_enabled:
            crop_to_oris = pose_data.crop_to_oris[b:b+bs]
            Ks_inv = pose_data.Ks_inv
          else:
            crop_to_oris = pose_data.tf_to_crops[b:b+bs].inverse().cuda()
            Ks_inv = pose_data.Ks[b:b+bs].inverse().cuda()
          uv_pred = transform_pts(uv_pred_crop, crop_to_oris)
          center_pred = torch.cat([uv_pred, torch.ones((len(rot_delta),1), dtype=torch.float, device='cuda')], dim=-1)
          center_pred = (Ks_inv@center_pred.reshape(len(rot_delta),3,1)).reshape(len(rot_delta),3) * z_pred.reshape(len(rot_delta),1)
          trans_delta = center_pred-pose_data.poseA[b:b+bs][...,:3,3]

        else:
          trans_delta = output["trans"]

        if self.cfg['rot_rep']=='axis_angle':
          rot_mat_delta = torch.tanh(output["rot"])*self.cfg['rot_normalizer']
          rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)
        elif self.cfg['rot_rep']=='6d':
          rot_mat_delta = rotation_6d_to_matrix(output['rot']).permute(0,2,1)
        else:
          raise RuntimeError

        if self.cfg['normalize_xyz']:
          trans_delta *= (mesh_diameter/2)

        B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA[b:b+bs], trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
        B_in_cams.append(B_in_cam)
        finish_cuda_stage_timing(timing, 'pose_update', stage_start)

      B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams),4,4)
      if capture_iteration_poses:
        self.last_iteration_poses.append(B_in_cams.detach().cpu().numpy().astype(np.float32))

    B_in_cams_out = B_in_cams@torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
    self.last_trans_update = trans_delta
    self.last_rot_update = rot_mat_delta
    torch.cuda.synchronize()
    for name, start_event, end_event in deferred_cuda_timing_events:
      timing[name] += start_event.elapsed_time(end_event) / 1000.0
    finalize_cuda_stage_timing(timing)
    timing['total'] = time.perf_counter() - total_start
    if self.render_profile_enabled:
      finalize_nvdiffrast_render_timing(timing)
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
    known_time = sum(timing[key] for key in ('crop_window', 'render', 'render_postprocess', 'warp', 'transform', 'input_pack', 'network_forward', 'output_convert', 'pose_update', 'empty_cache'))
    timing['other'] = max(timing['total'] - known_time, 0.0)
    timing['network_backend'] = next(iter(network_backends)) if len(network_backends) == 1 else 'mixed'
    timing['fallback_reason'] = '; '.join(sorted(fallback_reasons)) or None
    if network_backends == {'tensorrt'}:
      for name in ('encodeA', 'encodeAB', 'trans_head', 'rot_head'):
        timing[name] = None
    elif network_backends == {'pytorch'}:
      for name in ('layout_convert', 'dtype_convert', 'tensorrt_execute'):
        timing[name] = None
    timing.pop('_cuda_stage_events_enabled', None)
    self.last_timing = timing

    if get_vis:
      logging.info("get_vis...")
      canvas = []
      padding = 2
      pose_data = make_crop_data_batch(stage_input_size, torch.as_tensor(ob_centered_in_cams), mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        if pose_data.normalAs is not None:
          pass
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        row = cv_draw_text(row, text=f'id:{id}', uv_top_left=(10,10), color=(0,255,0), fontScale=0.5)
        canvas.append(row)
      canvas = make_grid_image(canvas, nrow=1, padding=padding, pad_value=255)

      pose_data = make_crop_data_batch(stage_input_size, B_in_cams, mesh_centered, rgb, depth, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, cfg=self.cfg, glctx=glctx, mesh_tensors=mesh_tensors, dataset=self.dataset, mesh_diameter=mesh_diameter)
      canvas_refined = []
      for id in range(0, len(B_in_cams)):
        rgbA_vis = (pose_data.rgbAs[id]*255).permute(1,2,0).data.cpu().numpy()
        rgbB_vis = (pose_data.rgbBs[id]*255).permute(1,2,0).data.cpu().numpy()
        row = [rgbA_vis, rgbB_vis]
        H,W = rgbA_vis.shape[:2]
        if pose_data.depthAs is not None:
          depthA = pose_data.depthAs[id].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.depthBs[id].data.cpu().numpy().reshape(H,W)
        elif pose_data.xyz_mapAs is not None:
          depthA = pose_data.xyz_mapAs[id][2].data.cpu().numpy().reshape(H,W)
          depthB = pose_data.xyz_mapBs[id][2].data.cpu().numpy().reshape(H,W)
        zmin = min(depthA.min(), depthB.min())
        zmax = max(depthA.max(), depthB.max())
        depthA_vis = depth_to_vis(depthA, zmin=zmin, zmax=zmax, inverse=False)
        depthB_vis = depth_to_vis(depthB, zmin=zmin, zmax=zmax, inverse=False)
        row += [depthA_vis, depthB_vis]
        row = make_grid_image(row, nrow=len(row), padding=padding, pad_value=255)
        canvas_refined.append(row)

      canvas_refined = make_grid_image(canvas_refined, nrow=1, padding=padding, pad_value=255)
      canvas = make_grid_image([canvas, canvas_refined], nrow=2, padding=padding, pad_value=255)
      torch.cuda.empty_cache()
      return B_in_cams_out, canvas

    return B_in_cams_out, None

