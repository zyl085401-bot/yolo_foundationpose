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
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
from network_modules import *
from Utils import *




class ScoreNetMultiPair(nn.Module):
  def __init__(self, cfg=None, c_in=4):
    super().__init__()
    self.cfg = cfg
    self._last_timing_events = {}
    if self.cfg.use_BN:
      norm_layer = nn.BatchNorm2d
    else:
      norm_layer = None

    self.encoderA = nn.Sequential(
      ConvBNReLU(C_in=c_in,C_out=64,kernel_size=7,stride=2, norm_layer=norm_layer),
      ConvBNReLU(C_in=64,C_out=128,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(128,128,bias=True, norm_layer=norm_layer),
    )

    self.encoderAB = nn.Sequential(
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(256,256,bias=True, norm_layer=norm_layer),
      ConvBNReLU(256,512,kernel_size=3,stride=2, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
      ResnetBasicBlock(512,512,bias=True, norm_layer=norm_layer),
    )

    embed_dim = 512
    num_heads = 4
    self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)
    self.att_cross = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, bias=True, batch_first=True)

    self.pos_embed = PositionalEmbedding(d_model=embed_dim, max_len=400)
    self.linear = nn.Linear(embed_dim, 1)


  def fuse_conv_batchnorm(self):
    """Fold inference BatchNorm2d parameters into their preceding convolutions."""
    if self.training:
      raise RuntimeError("Conv-BN fusion requires ScoreNetMultiPair.eval()")
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


  def extract_feat(self, A, B, timing_events=None):
    """
    @A: (B*L,C,H,W) L is num of pairs
    """
    bs = A.shape[0]  # B*L

    x = torch.cat([A,B], dim=0)
    if timing_events:
      timing_events['encoderA'][0].record()
    x = self.encoderA(x)
    if timing_events:
      timing_events['encoderA'][1].record()
    a = x[:bs]
    b = x[bs:]
    ab = torch.cat((a,b), dim=1).contiguous(memory_format=torch.channels_last)
    if timing_events:
      timing_events['encoderAB'][0].record()
    ab = self.encoderAB(ab)
    if timing_events:
      timing_events['encoderAB'][1].record()
    ab = self.pos_embed(ab.reshape(bs, ab.shape[1], -1).permute(0,2,1))
    if timing_events:
      timing_events['self_attention'][0].record()
    ab, _ = self.att(ab, ab, ab)
    if timing_events:
      timing_events['self_attention'][1].record()
    return ab.mean(dim=1).reshape(bs,-1)


  def forward(self, A, B, L):
    """
    @A: (B*L,C,H,W) L is num of pairs
    @L: num of pairs
    """
    output = {}
    bs = A.shape[0]//L
    timing_events = {
      name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
      for name in ('network_forward', 'encoderA', 'encoderAB', 'self_attention', 'cross_attention', 'linear')
    } if A.is_cuda else {}
    if timing_events:
      timing_events['network_forward'][0].record()
    feats = self.extract_feat(A, B, timing_events=timing_events)   #(B*L, C)
    x = feats.reshape(bs,L,-1)
    if timing_events:
      timing_events['cross_attention'][0].record()
    x, _ = self.att_cross(x, x, x)
    if timing_events:
      timing_events['cross_attention'][1].record()

    if timing_events:
      timing_events['linear'][0].record()
    output['score_logit'] = self.linear(x).reshape(bs,L)  # (B,L)
    if timing_events:
      timing_events['linear'][1].record()
      timing_events['network_forward'][1].record()
    self._last_timing_events = timing_events

    return output


  def collect_last_cuda_timing(self):
    """Return synchronized CUDA module timings in seconds."""
    return {
      name: start.elapsed_time(end) / 1000.0
      for name, (start, end) in self._last_timing_events.items()
    }
