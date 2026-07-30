from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import onnx
import torch
from omegaconf import OmegaConf


FOUNDATIONPOSE_ROOT = Path(__file__).resolve().parents[1]
if str(FOUNDATIONPOSE_ROOT) not in sys.path:
  sys.path.insert(0, str(FOUNDATIONPOSE_ROOT))

from learning.models.refine_network import RefineNet  # noqa: E402
from learning.models.score_network import ScoreNetMultiPair  # noqa: E402


NETWORK_DEFAULTS = {
    'refiner': {
        'run_name': '2023-10-28-18-33-37',
        'batch': 12,
        'outputs': ('trans', 'rot'),
    },
    'scorer': {
        'run_name': '2024-01-11-20-02-45',
        'batch': 6,
        'outputs': ('score_logit',),
    },
}


class RefinerExportWrapper(torch.nn.Module):
  def __init__(self, model: RefineNet):
    super().__init__()
    self.model = model

  def forward(self, A: torch.Tensor, B: torch.Tensor):
    output = self.model(A, B)
    return output['trans'], output['rot']


class ScorerExportWrapper(torch.nn.Module):
  def __init__(self, model: ScoreNetMultiPair, candidate_count: int, dynamic_batch: bool = False):
    super().__init__()
    self.model = model
    self.candidate_count = candidate_count
    self.dynamic_batch = dynamic_batch

  def forward(self, A: torch.Tensor, B: torch.Tensor):
    candidate_count = A.shape[0] if self.dynamic_batch else self.candidate_count
    return self.model(A, B, L=candidate_count)['score_logit']


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def load_checkpoint(path: Path):
  checkpoint = torch.load(path, map_location='cpu', weights_only=False)
  return checkpoint.get('model', checkpoint)


def build_wrapper(network: str, config, checkpoint_path: Path, batch: int, dynamic_batch: bool) -> torch.nn.Module:
  if network == 'refiner':
    model = RefineNet(cfg=config, c_in=config.c_in).cpu().eval()
    model.load_state_dict(load_checkpoint(checkpoint_path))
    model.fuse_conv_batchnorm()
    return RefinerExportWrapper(model).eval()

  model = ScoreNetMultiPair(cfg=config, c_in=config.c_in).cpu().eval()
  model.load_state_dict(load_checkpoint(checkpoint_path))
  model.fuse_conv_batchnorm()
  return ScorerExportWrapper(model, candidate_count=batch, dynamic_batch=dynamic_batch).eval()


def main() -> None:
  parser = argparse.ArgumentParser(description='Export fixed- or dynamic-batch FoundationPose ONNX models.')
  parser.add_argument('--network', choices=sorted(NETWORK_DEFAULTS), required=True)
  parser.add_argument('--batch', type=int)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--opset', type=int, default=17)
  parser.add_argument('--dynamic-batch', action='store_true')
  parser.add_argument('--min-batch', type=int)
  parser.add_argument('--max-batch', type=int)
  parser.add_argument(
      '--input-size',
      type=int,
      nargs=2,
      metavar=('HEIGHT', 'WIDTH'),
      help='Override config.input_resize while keeping spatial dimensions fixed in the exported ONNX model.',
  )
  args = parser.parse_args()

  defaults = NETWORK_DEFAULTS[args.network]
  batch = int(args.batch or defaults['batch'])
  if batch < 1:
    raise ValueError('batch must be positive')
  min_batch = int(args.min_batch or batch)
  max_batch = int(args.max_batch or batch)
  if args.dynamic_batch:
    if not 1 <= min_batch <= batch <= max_batch:
      raise ValueError(f'Expected 1 <= min_batch <= batch <= max_batch, got {min_batch}, {batch}, {max_batch}')
  elif args.min_batch is not None or args.max_batch is not None:
    raise ValueError('--min-batch and --max-batch require --dynamic-batch')

  run_dir = FOUNDATIONPOSE_ROOT / 'weights' / defaults['run_name']
  config_path = run_dir / 'config.yml'
  checkpoint_path = run_dir / 'model_best.pth'
  config = OmegaConf.load(config_path)
  wrapper = build_wrapper(args.network, config, checkpoint_path, batch, args.dynamic_batch)

  channels = int(config.c_in)
  height, width = (
      tuple(args.input_size)
      if args.input_size is not None
      else tuple(int(value) for value in config.input_resize)
  )
  if height <= 0 or width <= 0:
    raise ValueError(f'input size must be positive, got {(height, width)}')
  token_count = ((height + 7) // 8) * ((width + 7) // 8)
  if token_count > 400:
    raise ValueError(f'input size {(height, width)} produces {token_count} tokens, exceeding positional embedding limit 400')
  torch.manual_seed(20260723)
  A = torch.randn(batch, channels, height, width, dtype=torch.float32)
  B = torch.randn_like(A)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  previous_fastpath = torch.backends.mha.get_fastpath_enabled()
  torch.backends.mha.set_fastpath_enabled(False)
  dynamic_axes = None
  if args.dynamic_batch:
    dynamic_axes = {
        'A': {0: 'candidate_count'},
        'B': {0: 'candidate_count'},
    }
    output_axis = 1 if args.network == 'scorer' else 0
    for output_name in defaults['outputs']:
      dynamic_axes[output_name] = {output_axis: 'candidate_count'}
  try:
    with torch.inference_mode():
      torch.onnx.export(
          wrapper,
          (A, B),
          args.output,
          input_names=('A', 'B'),
          output_names=defaults['outputs'],
          opset_version=args.opset,
          do_constant_folding=True,
          export_params=True,
          dynamo=False,
          dynamic_axes=dynamic_axes,
      )
  finally:
    torch.backends.mha.set_fastpath_enabled(previous_fastpath)

  model = onnx.load(args.output)
  onnx.checker.check_model(model, full_check=True)
  inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
  onnx.save(inferred, args.output)

  with torch.inference_mode():
    reference_outputs = wrapper(A, B)
  if not isinstance(reference_outputs, tuple):
    reference_outputs = (reference_outputs,)

  metadata = {
      'network': args.network,
      'batch': batch,
      'input_shape': [batch, channels, height, width],
      'input_size': [height, width],
      'input_size_source': 'command_line' if args.input_size is not None else 'model_config',
      'input_dtype': 'float32',
      'outputs': {
        name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
        for name, tensor in zip(defaults['outputs'], reference_outputs)
      },
      'opset': args.opset,
      'torch_version': torch.__version__,
      'onnx_version': onnx.__version__,
      'checkpoint': str(checkpoint_path.relative_to(FOUNDATIONPOSE_ROOT.parent)),
      'checkpoint_sha256': file_sha256(checkpoint_path),
      'config': str(config_path.relative_to(FOUNDATIONPOSE_ROOT.parent)),
      'config_sha256': file_sha256(config_path),
      'onnx_sha256': file_sha256(args.output),
      'attention_fastpath_during_export': False,
      'fixed_shape': not args.dynamic_batch,
      'batch_profile': {
        'min': min_batch,
        'opt': batch,
        'max': max_batch,
      },
  }
  metadata_path = args.output.with_suffix(args.output.suffix + '.json')
  metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
