#!/usr/bin/env python3
"""Compare a captured original road tensor with the untapped LiteRT CPU graph."""
import argparse
import json
from pathlib import Path

from ai_edge_litert.interpreter import Interpreter
import numpy as np

from validate_litert_road_host import digest


def trace(model, reference, fixtures, output, dense_fixtures=None):
  if output.exists():
    raise ValueError('Use a new report path')
  manifest = json.loads((fixtures / 'manifest.json').read_text())
  case = manifest['cases']['trace_0']
  if case['original_tensor'] != 'model/linear' or case['seed'] != 42:
    raise ValueError('Unexpected original trace contract')
  runtime = Interpreter(model_path=str(model), num_threads=1, experimental_preserve_all_tensors=True)
  runtime.allocate_tensors()
  input_path = reference / 'driving_supercombo-inputs.npz'
  axes = {'new_img': (0, 2, 3, 1), 'state_img_q': (0, 2, 3, 1),
          'state_desire_q': (0, 2, 1), 'state_feat_q': (0, 2, 1)}
  with np.load(input_path) as captured:
    for tensor in runtime.get_input_details():
      name = tensor['name']
      value = captured[name]
      if name == 'state_img_q':
        value = value.reshape(2, 30, 128, 256).astype(np.float32)
      if name in axes:
        value = value.transpose(axes[name])
      if name == 'new_img':
        value = value.astype(np.float32)
      value = np.ascontiguousarray(value)
      if tuple(value.shape) != tuple(tensor['shape']) or value.dtype != tensor['dtype']:
        raise ValueError('Input contract mismatch: ' + name)
      if name in case['inputs']:
        expected = fixtures / 'trace_0' / case['inputs'][name]['file']
        if digest(expected) != case['inputs'][name]['sha256'] or value.tobytes() != expected.read_bytes():
          raise ValueError('Trace input does not match captured reference: ' + name)
      runtime.set_tensor(tensor['index'], value)
  runtime.invoke()
  # The converter's correspondence report names model/linear, but its final
  # float32 model removes that cast. Compare the surviving FC value and a
  # diagnostic half-rounding separately; neither is an actual saved cast.
  matches = [item for item in runtime.get_tensor_details() if item['name'] == 'model/linear_fc_float32']
  if len(matches) != 1:
    raise ValueError('Unable to uniquely identify model/linear FC tensor')
  actual = runtime.get_tensor(matches[0]['index']).reshape(-1)
  golden_path = fixtures / 'trace_0' / 'reference.npz'
  with np.load(golden_path) as golden:
    expected = golden['trace_flat'].reshape(-1)
  if expected.shape != actual.shape:
    raise ValueError('Trace shape mismatch')
  close = np.isclose(actual, expected, rtol=1e-3, atol=1e-4)
  rounded = actual.astype(np.float16).astype(np.float32)
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'execution': 'host CPU diagnostic',
            'tensor': 'model/linear_fc_float32', 'reference_tensor': 'model/linear',
            'boundary': 'FC output before expected half rounding; post-rounding is diagnostic only',
            'model_sha256': digest(model), 'inputs_sha256': digest(input_path),
            'trace_manifest_sha256': digest(fixtures / 'manifest.json'), 'reference_sha256': digest(golden_path),
            'mismatches': int((~close).sum()), 'post_half_round_mismatches': int((~np.isclose(rounded, expected, rtol=1e-3, atol=1e-4)).sum()), 'count': actual.size,
            'nonfinite': int((~np.isfinite(actual)).sum()),
            'max_abs_error': float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))}
  if dense_fixtures is not None:
    dense_manifest = json.loads((dense_fixtures / 'manifest.json').read_text())
    capture = dense_manifest['cases']['dense']['inputs']['input']
    capture_path = dense_fixtures / 'dense' / capture['file']
    if digest(capture_path) != capture['sha256']:
      raise ValueError('Dense input capture hash mismatch')
    producers = [op for op in runtime._get_ops_details()
                 if op['op_name'] == 'FULLY_CONNECTED' and matches[0]['index'] in op['outputs']]
    if len(producers) != 1 or producers[0]['op_name'] != 'FULLY_CONNECTED':
      raise ValueError('Unable to uniquely identify the dense input')
    dense_input = runtime.get_tensor(int(producers[0]['inputs'][0])).reshape(-1)
    original_input = np.fromfile(capture_path, dtype=np.float32)
    if dense_input.shape != original_input.shape:
      raise ValueError('Dense input shape mismatch')
    report['dense_input'] = {
      'capture_sha256': digest(capture_path), 'count': dense_input.size,
      'mismatches': int((~np.isclose(dense_input, original_input, rtol=1e-3, atol=1e-4)).sum()),
      'nonfinite': int((~np.isfinite(dense_input)).sum()),
      'max_abs_error': float(np.max(np.abs(dense_input.astype(np.float64) - original_input.astype(np.float64))))}
  output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))
  return report


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('model', 'reference', 'fixtures', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--dense-fixtures', type=Path)
  args = parser.parse_args()
  result = trace(args.model, args.reference, args.fixtures, args.output, args.dense_fixtures)
  raise SystemExit(bool(result['mismatches'] or result['nonfinite']))
