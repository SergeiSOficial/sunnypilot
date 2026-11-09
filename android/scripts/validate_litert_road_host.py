#!/usr/bin/env python3
"""Compare an onnx2tf road export with original captures; CPU diagnostic only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ai_edge_litert.interpreter import Interpreter


def digest(path):
  with path.open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def validate(model, reference, output, packed_history=False):
  if output.exists():
    raise ValueError('Use a new report path to preserve earlier evidence')
  runtime = Interpreter(model_path=str(model), num_threads=1)
  runtime.allocate_tensors()
  # onnx2tf 2.6.5 moves channel dimension to the end for these named inputs.
  axes = {'new_img': (0, 2, 3, 1), 'state_img_q': (0, 2, 3, 4, 1),
          'state_desire_q': (0, 2, 1), 'state_feat_q': (0, 2, 1)}
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'execution': 'host CPU diagnostic',
            'model_sha256': digest(model), 'packed_history': packed_history,
            'rtol': 1e-3, 'atol': 1e-4, 'cases': {}}
  for suffix in ('', '-case-7', '-case-123'):
    inp = reference / ('driving_supercombo' + suffix + '-inputs.npz')
    gold = reference / ('driving_supercombo' + suffix + '-outputs.npz')
    case = {'input_sha256': digest(inp), 'reference_sha256': digest(gold), 'outputs': {}}
    with np.load(inp) as source, np.load(gold) as expected:
      for t in runtime.get_input_details():
        name = t['name']
        data = source[name]
        if packed_history and name == 'state_img_q':
          data = data.reshape(2, 30, 128, 256).astype(np.float32)
        if name in axes:
          data = data.transpose((0, 2, 3, 1) if packed_history and name == 'state_img_q' else axes[name])
        if packed_history and name == 'new_img':
          data = data.astype(np.float32)
        if tuple(data.shape) != tuple(t['shape']) or data.dtype != t['dtype']:
          raise ValueError(f'Input contract mismatch for {name}')
        runtime.set_tensor(t['index'], data)
      runtime.invoke()
      for t in runtime.get_output_details():
        name = t['name']
        actual, target = runtime.get_tensor(t['index']), expected[name]
        if packed_history and name == 'next_state_img_q':
          actual = actual.reshape(target.shape)
        if actual.shape != target.shape:
          raise ValueError(f'Output shape mismatch for {name}')
        matches = actual == target if name == 'next_state_img_q' else np.isclose(actual, target, rtol=1e-3, atol=1e-4)
        error = np.abs(actual.astype(np.float64) - target.astype(np.float64))
        case['outputs'][name] = {'mismatches': int((~matches).sum()),
          'nonfinite': int((~np.isfinite(actual)).sum()), 'max_abs_error': float(error.max())}
    report['cases'][suffix or 'seed42'] = case
  output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
  return report


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('model', 'reference', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--packed-history', action='store_true', help='Use the proven packed image-history and float transport contract')
  args = parser.parse_args()
  result = validate(args.model, args.reference, args.output, packed_history=args.packed_history)
  print(json.dumps(result, indent=2))
  if any(v['mismatches'] or v['nonfinite'] for case in result['cases'].values() for v in case['outputs'].values()):
    raise SystemExit(1)
