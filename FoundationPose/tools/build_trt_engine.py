from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tensorrt as trt
import torch


LOGGER = trt.Logger(trt.Logger.INFO)


def file_sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open('rb') as file:
    for chunk in iter(lambda: file.read(1024 * 1024), b''):
      digest.update(chunk)
  return digest.hexdigest()


def main() -> None:
  parser = argparse.ArgumentParser(description='Build a fixed- or dynamic-batch TensorRT engine from ONNX.')
  parser.add_argument('--onnx', type=Path, required=True)
  parser.add_argument('--engine', type=Path, required=True)
  parser.add_argument('--workspace-gib', type=float, default=2.0)
  parser.add_argument('--optimization-level', type=int, default=5, choices=range(0, 6))
  parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=True)
  parser.add_argument('--min-batch', type=int)
  parser.add_argument('--opt-batch', type=int)
  parser.add_argument('--max-batch', type=int)
  parser.add_argument(
      '--fp32-layer-types',
      nargs='*',
      default=(),
      help='TensorRT LayerType names to keep in FP32 when --fp16 is enabled, for example MATRIX_MULTIPLY SOFTMAX.',
  )
  args = parser.parse_args()

  if not args.onnx.is_file():
    raise FileNotFoundError(args.onnx)
  builder = trt.Builder(LOGGER)
  network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
  parser_trt = trt.OnnxParser(network, LOGGER)
  if not parser_trt.parse_from_file(str(args.onnx)):
    errors = '\n'.join(str(parser_trt.get_error(index)) for index in range(parser_trt.num_errors))
    raise RuntimeError(f'TensorRT failed to parse {args.onnx}:\n{errors}')

  config = builder.create_builder_config()
  config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gib * 1024**3))
  config.builder_optimization_level = args.optimization_level
  dynamic_inputs = [network.get_input(index) for index in range(network.num_inputs) if -1 in tuple(network.get_input(index).shape)]
  profile_metadata = None
  if dynamic_inputs:
    if args.min_batch is None or args.opt_batch is None or args.max_batch is None:
      raise ValueError('Dynamic ONNX inputs require --min-batch, --opt-batch, and --max-batch')
    if not 1 <= args.min_batch <= args.opt_batch <= args.max_batch:
      raise ValueError('Expected 1 <= min_batch <= opt_batch <= max_batch')
    profile = builder.create_optimization_profile()
    profile_metadata = {}
    for tensor in dynamic_inputs:
      base_shape = tuple(tensor.shape)
      if base_shape[0] != -1 or any(dimension < 0 for dimension in base_shape[1:]):
        raise ValueError(f'Only a dynamic leading batch is supported, got {tensor.name}: {base_shape}')
      shapes = {
          'min': (args.min_batch, *base_shape[1:]),
          'opt': (args.opt_batch, *base_shape[1:]),
          'max': (args.max_batch, *base_shape[1:]),
      }
      profile.set_shape(tensor.name, shapes['min'], shapes['opt'], shapes['max'])
      profile_metadata[tensor.name] = {name: list(shape) for name, shape in shapes.items()}
    config.add_optimization_profile(profile)
  elif any(value is not None for value in (args.min_batch, args.opt_batch, args.max_batch)):
    raise ValueError('Batch profile arguments were supplied for a fixed-shape ONNX model')
  if args.fp16:
    if not builder.platform_has_fast_fp16:
      raise RuntimeError('TensorRT reports that this platform has no fast FP16 support')
    config.set_flag(trt.BuilderFlag.FP16)

  fp32_layer_types = set()
  for name in args.fp32_layer_types:
    normalized_name = name.strip().upper()
    if not hasattr(trt.LayerType, normalized_name):
      valid_names = sorted(name for name in dir(trt.LayerType) if name.isupper())
      raise ValueError(f'Unknown TensorRT LayerType {name!r}; valid names: {valid_names}')
    fp32_layer_types.add(getattr(trt.LayerType, normalized_name))
  constrained_layers = []
  if fp32_layer_types:
    if not args.fp16:
      raise ValueError('--fp32-layer-types is only meaningful with --fp16')
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    for layer in network:
      if layer.type not in fp32_layer_types:
        continue
      layer.precision = trt.float32
      for output_index in range(layer.num_outputs):
        output = layer.get_output(output_index)
        if not output.is_shape_tensor:
          layer.set_output_type(output_index, trt.float32)
      constrained_layers.append({'name': layer.name, 'type': str(layer.type)})

  serialized_engine = builder.build_serialized_network(network, config)
  if serialized_engine is None:
    raise RuntimeError('TensorRT returned no serialized engine')
  args.engine.parent.mkdir(parents=True, exist_ok=True)
  args.engine.write_bytes(serialized_engine)

  runtime = trt.Runtime(LOGGER)
  engine = runtime.deserialize_cuda_engine(serialized_engine)
  if engine is None:
    raise RuntimeError('Built engine could not be deserialized')
  io_tensors = []
  for index in range(engine.num_io_tensors):
    name = engine.get_tensor_name(index)
    io_tensors.append({
        'name': name,
        'mode': str(engine.get_tensor_mode(name)),
        'dtype': str(engine.get_tensor_dtype(name)),
        'shape': list(engine.get_tensor_shape(name)),
    })

  onnx_metadata_path = args.onnx.with_suffix(args.onnx.suffix + '.json')
  onnx_metadata = json.loads(onnx_metadata_path.read_text(encoding='utf-8')) if onnx_metadata_path.is_file() else {}
  metadata = {
      **onnx_metadata,
      'engine': str(args.engine),
      'engine_sha256': file_sha256(args.engine),
      'engine_size_bytes': args.engine.stat().st_size,
      'tensorrt_version': trt.__version__,
      'torch_version_at_build': torch.__version__,
      'cuda_device_name': torch.cuda.get_device_name(0),
      'cuda_compute_capability': list(torch.cuda.get_device_capability(0)),
      'fp16': args.fp16,
      'fp32_layer_types': sorted(name.strip().upper() for name in args.fp32_layer_types),
      'fp32_constrained_layers': constrained_layers,
      'workspace_gib': args.workspace_gib,
      'builder_optimization_level': args.optimization_level,
      'optimization_profile': profile_metadata,
      'io_tensors': io_tensors,
  }
  metadata_path = args.engine.with_suffix(args.engine.suffix + '.json')
  metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n', encoding='utf-8')
  print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
