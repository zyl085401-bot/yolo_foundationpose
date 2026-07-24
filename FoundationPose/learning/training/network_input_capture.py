from __future__ import annotations

import json
import os
import threading
from collections import Counter
from typing import Any

import torch


class NetworkInputCapture:
  """Optionally persist a bounded set of real network inputs for backend validation."""

  def __init__(self, network_name: str, config: dict[str, Any] | None = None):
    config = dict(config or {})
    self.network_name = network_name
    enabled_override = os.getenv('FOUNDATIONPOSE_CAPTURE_NETWORK_INPUTS')
    self.enabled = bool(config.get('enabled', False)) if enabled_override is None else enabled_override.strip().lower() in ('1', 'true', 'yes', 'on')
    output_dir = os.getenv('FOUNDATIONPOSE_CAPTURE_OUTPUT_DIR', config.get('output_dir', 'network_input_capture'))
    max_samples = os.getenv('FOUNDATIONPOSE_CAPTURE_MAX_SAMPLES', config.get('max_samples_per_signature', 1))
    self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
    self.max_samples_per_signature = max(1, int(max_samples))
    self._counts: Counter[str] = Counter()
    self._observations: Counter[str] = Counter()
    self._lock = threading.Lock()
    if self.enabled:
      os.makedirs(self.output_dir, exist_ok=True)

  def capture(self, stage: str, A: torch.Tensor, B: torch.Tensor, output: dict[str, torch.Tensor], **metadata: Any) -> str | None:
    if not self.enabled:
      return None

    signature = self._signature(stage, A, metadata)
    with self._lock:
      self._observations[signature] += 1
      sample_index = self._counts[signature]
      if sample_index >= self.max_samples_per_signature:
        self._write_manifest()
        return None
      self._counts[signature] += 1

    sample = {
      'network': self.network_name,
      'stage': stage,
      'A': self._cpu_nchw(A),
      'B': self._cpu_nchw(B),
      'output': {
        name: tensor.detach().to(device='cpu', dtype=torch.float32).contiguous()
        for name, tensor in output.items()
        if torch.is_tensor(tensor)
      },
      'input_metadata': {
        'A_shape': list(A.shape),
        'B_shape': list(B.shape),
        'A_dtype': str(A.dtype),
        'B_dtype': str(B.dtype),
        'A_stride': list(A.stride()),
        'B_stride': list(B.stride()),
        'A_contiguous': bool(A.is_contiguous()),
        'B_contiguous': bool(B.is_contiguous()),
        'A_channels_last': bool(A.is_contiguous(memory_format=torch.channels_last)),
        'B_channels_last': bool(B.is_contiguous(memory_format=torch.channels_last)),
      },
      'output_metadata': {
        name: {
          'shape': list(tensor.shape),
          'dtype': str(tensor.dtype),
          'stride': list(tensor.stride()),
        }
        for name, tensor in output.items()
        if torch.is_tensor(tensor)
      },
      'metadata': self._json_safe(metadata),
    }
    filename = f'{self.network_name}_{signature}_{sample_index:02d}.pt'
    path = os.path.join(self.output_dir, filename)
    torch.save(sample, path)
    self._write_manifest()
    return path

  def _signature(self, stage: str, A: torch.Tensor, metadata: dict[str, Any]) -> str:
    candidate_count = int(A.shape[0])
    parts = [self._sanitize(stage), f'n{candidate_count}']
    if metadata.get('L') is not None:
      parts.append(f'L{int(metadata["L"])}')
    return '_'.join(parts)

  def _write_manifest(self) -> None:
    manifest = {
      'network': self.network_name,
      'max_samples_per_signature': self.max_samples_per_signature,
      'captured': dict(sorted(self._counts.items())),
      'observations': dict(sorted(self._observations.items())),
    }
    path = os.path.join(self.output_dir, f'{self.network_name}_manifest.json')
    temporary_path = f'{path}.tmp'
    with open(temporary_path, 'w', encoding='utf-8') as file:
      json.dump(manifest, file, indent=2, sort_keys=True)
      file.write('\n')
    os.replace(temporary_path, path)

  @staticmethod
  def _cpu_nchw(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device='cpu', dtype=torch.float32).contiguous()

  @staticmethod
  def _sanitize(value: str) -> str:
    return ''.join(character if character.isalnum() or character in ('-', '_') else '_' for character in value)

  @classmethod
  def _json_safe(cls, value: Any) -> Any:
    if isinstance(value, dict):
      return {str(key): cls._json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
      return [cls._json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
      return value
    return str(value)
