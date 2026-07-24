from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import tensorrt as trt
import torch


_TRT_TO_TORCH_DTYPE = {
    trt.float16: torch.float16,
    trt.float32: torch.float32,
    trt.int8: torch.int8,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}


class TensorRTShapeError(ValueError):
  pass


class TensorRTEngineRunner:
  """TensorRT 10 runner backed directly by PyTorch CUDA tensors and streams."""

  def __init__(self, engine_path: str, expected_network: str | None = None, expected_checkpoint_path: str | None = None):
    self.engine_path = Path(engine_path).expanduser().resolve()
    if not self.engine_path.is_file():
      raise FileNotFoundError(self.engine_path)
    self.metadata_path = self.engine_path.with_suffix(self.engine_path.suffix + '.json')
    if not self.metadata_path.is_file():
      raise FileNotFoundError(f'Missing TensorRT metadata: {self.metadata_path}')
    self.metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))
    self._validate_metadata(expected_network, expected_checkpoint_path)

    self.logger = trt.Logger(trt.Logger.WARNING)
    self.runtime = trt.Runtime(self.logger)
    self.engine = self.runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
    if self.engine is None:
      raise RuntimeError(f'Failed to deserialize TensorRT engine: {self.engine_path}')
    self.context = self.engine.create_execution_context()
    if self.context is None:
      raise RuntimeError(f'Failed to create TensorRT execution context: {self.engine_path}')

    self.input_names = []
    self.output_names = []
    for index in range(self.engine.num_io_tensors):
      name = self.engine.get_tensor_name(index)
      mode = self.engine.get_tensor_mode(name)
      if mode == trt.TensorIOMode.INPUT:
        self.input_names.append(name)
      elif mode == trt.TensorIOMode.OUTPUT:
        self.output_names.append(name)
    self._output_buffers: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
    self._execution_streams: dict[torch.device, torch.cuda.Stream] = {}
    self._lock = threading.Lock()
    self._last_timing_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None

  def __call__(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    missing = set(self.input_names) - set(inputs)
    extra = set(inputs) - set(self.input_names)
    if missing or extra:
      raise ValueError(f'TensorRT bindings mismatch: missing={sorted(missing)}, extra={sorted(extra)}')

    with self._lock:
      device = self._validate_inputs(inputs)
      for name in self.input_names:
        tensor = inputs[name]
        engine_shape = tuple(self.engine.get_tensor_shape(name))
        if any(dimension < 0 for dimension in engine_shape):
          if not self.context.set_input_shape(name, tuple(tensor.shape)):
            raise TensorRTShapeError(f'TensorRT rejected {name} shape {tuple(tensor.shape)}')
        elif tuple(tensor.shape) != engine_shape:
          raise TensorRTShapeError(f'{name} expects fixed shape {engine_shape}, got {tuple(tensor.shape)}')
        self.context.set_tensor_address(name, tensor.data_ptr())

      outputs: dict[str, torch.Tensor] = {}
      for name in self.output_names:
        shape = tuple(self.context.get_tensor_shape(name))
        if any(dimension < 0 for dimension in shape):
          raise TensorRTShapeError(f'Unresolved output shape for {name}: {shape}')
        key = (name, shape)
        tensor = self._output_buffers.get(key)
        dtype = self._torch_dtype(name)
        if tensor is None or tensor.device != device or tensor.dtype != dtype:
          tensor = torch.empty(shape, device=device, dtype=dtype)
          self._output_buffers[key] = tensor
        self.context.set_tensor_address(name, tensor.data_ptr())
        outputs[name] = tensor

      producer_stream = torch.cuda.current_stream(device=device)
      stream = self._execution_streams.get(device)
      if stream is None:
        stream = torch.cuda.Stream(device=device)
        self._execution_streams[device] = stream
      stream.wait_stream(producer_stream)
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record(stream)
      if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
        raise RuntimeError(f'TensorRT execution failed: {self.engine_path}')
      end.record(stream)
      for tensor in (*inputs.values(), *outputs.values()):
        tensor.record_stream(stream)
      producer_stream.wait_stream(stream)
      self._last_timing_events = (start, end)
      return outputs

  def collect_last_cuda_timing(self) -> float | None:
    if self._last_timing_events is None:
      return None
    start, end = self._last_timing_events
    return start.elapsed_time(end) / 1000.0

  def _validate_inputs(self, inputs: dict[str, torch.Tensor]) -> torch.device:
    devices = {tensor.device for tensor in inputs.values()}
    if len(devices) != 1:
      raise ValueError(f'All TensorRT inputs must share one device, got {sorted(map(str, devices))}')
    device = next(iter(devices))
    if device.type != 'cuda':
      raise ValueError(f'TensorRT inputs must be CUDA tensors, got {device}')
    for name, tensor in inputs.items():
      expected_dtype = self._torch_dtype(name)
      if tensor.dtype != expected_dtype:
        raise TypeError(f'{name} expects {expected_dtype}, got {tensor.dtype}')
      if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous NCHW memory')
    return device

  def _torch_dtype(self, name: str) -> torch.dtype:
    dtype = self.engine.get_tensor_dtype(name)
    if dtype not in _TRT_TO_TORCH_DTYPE:
      raise TypeError(f'Unsupported TensorRT dtype for {name}: {dtype}')
    return _TRT_TO_TORCH_DTYPE[dtype]

  def _validate_metadata(self, expected_network: str | None, expected_checkpoint_path: str | None) -> None:
    if expected_network is not None and self.metadata.get('network') != expected_network:
      raise RuntimeError(f'Engine network mismatch: expected {expected_network}, got {self.metadata.get("network")}')
    if self.metadata.get('engine_sha256') != self._file_sha256(self.engine_path):
      raise RuntimeError(f'Engine SHA256 mismatch: {self.engine_path}')
    if self.metadata.get('tensorrt_version') != trt.__version__:
      raise RuntimeError(f'TensorRT version mismatch: engine={self.metadata.get("tensorrt_version")}, runtime={trt.__version__}')
    expected_capability = list(torch.cuda.get_device_capability(0))
    if self.metadata.get('cuda_compute_capability') != expected_capability:
      raise RuntimeError(f'Compute capability mismatch: engine={self.metadata.get("cuda_compute_capability")}, runtime={expected_capability}')
    if expected_checkpoint_path is not None:
      checkpoint_path = Path(expected_checkpoint_path).expanduser().resolve()
      checkpoint_hash = self._file_sha256(checkpoint_path)
      if self.metadata.get('checkpoint_sha256') != checkpoint_hash:
        raise RuntimeError(f'Checkpoint SHA256 mismatch: {checkpoint_path}')

  @staticmethod
  def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
      for chunk in iter(lambda: file.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()
