# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
sys.path.append(f'{code_dir}/../../../../')
from Utils import *
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from functools import partial
from network_modules import *
from Utils import *



class RefineNet(nn.Module):
  def __init__(self, cfg=None, c_in=4, n_view=1):
    super().__init__()
    self.cfg = cfg
    self._last_timing_events = {}
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
      norm_layer1d = nn.BatchNorm1d
    else:
      norm_layer = None
      norm_layer1d = None

    self.encodeA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encodeAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)

    self.trans_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, 3),
    )

    if self.cfg['rot_rep']=='axis_angle':
      rot_out_dim = 3
    elif self.cfg['rot_rep']=='6d':
      rot_out_dim = 6
    else:
      raise RuntimeError
    self.rot_head = nn.Sequential(
      nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True),
		  nn.Linear(512, rot_out_dim),
    )


  def fuse_conv_batchnorm(self):
    """Fold inference BatchNorm2d parameters into their preceding convolutions."""
    if self.training:
      raise RuntimeError("Conv-BN fusion requires RefineNet.eval()")
    if getattr(self, '_conv_bn_fused', False):
      return self

    from torch.nn.utils.fusion import fuse_conv_bn_eval

    for module in self.modules():
      if isinstance(module, ConvBNReLU) and len(module.net) >= 2 and isinstance(module.net[1], nn.BatchNorm2d):
        module.net[0] = fuse_conv_bn_eval(module.net[0], module.net[1])
        module.net[1] = nn.Identity()
      elif isinstance(module, ResnetBasicBlock) and module.norm_layer is not None:
        module.conv1 = fuse_conv_bn_eval(module.conv1, module.bn1)
        module.bn1 = nn.Identity()
        module.conv2 = fuse_conv_bn_eval(module.conv2, module.bn2)
        module.bn2 = nn.Identity()
        module.norm_layer = None

    self._conv_bn_fused = True
    return self


  def forward(self, A, B):
    """
    @A: (B,C,H,W)
    """
    bs = A.shape[0]
    output = {}
    timing_events = {
      name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
      for name in ('encodeA', 'encodeAB', 'trans_head', 'rot_head')
    } if A.is_cuda else {}

    x = torch.cat([A,B], dim=0)
    if timing_events:
      timing_events['encodeA'][0].record()
    x = self.encodeA(x)
    if timing_events:
      timing_events['encodeA'][1].record()
    a = x[:bs]
    b = x[bs:]

    ab = torch.cat((a,b),1).contiguous(memory_format=torch.channels_last)
    if timing_events:
      timing_events['encodeAB'][0].record()
    ab = self.encodeAB(ab)  #(B,C,H,W)
    if timing_events:
      timing_events['encodeAB'][1].record()

    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))

    if timing_events:
      timing_events['trans_head'][0].record()
    output['trans'] = self.trans_head(ab).mean(dim=1)
    if timing_events:
      timing_events['trans_head'][1].record()
      timing_events['rot_head'][0].record()
    output['rot'] = self.rot_head(ab).mean(dim=1)
    if timing_events:
      timing_events['rot_head'][1].record()
    self._last_timing_events = timing_events

    return output


  def extract_shared_feature(self, A, B):
    """Return the mean-pooled shared tokens without changing pose-head outputs."""
    bs = A.shape[0]
    x = self.encodeA(torch.cat([A, B], dim=0))
    ab = torch.cat((x[:bs], x[bs:]), 1).contiguous(memory_format=torch.channels_last)
    ab = self.encodeAB(ab)
    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0, 2, 1))
    return ab.mean(dim=1)


  def collect_last_cuda_timing(self):
    """Return synchronized CUDA module timings in seconds."""
    return {
      name: start.elapsed_time(end) / 1000.0
      for name, (start, end) in self._last_timing_events.items()
    }
