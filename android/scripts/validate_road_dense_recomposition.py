#!/usr/bin/env python3
"""Recompose a saved GPU dense result with host road sections; never qualifies a backend."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from decompose_road import ROOT, Recompose, digest
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from validate_road_dense_opencl import fixture_paths


def validate_case(sections_dir, fixtures, device_output, seed):
  manifest = json.loads((sections_dir/'manifest.json').read_text())
  receipt = json.loads((device_output/'results.json').read_text())
  fixture_paths(fixtures)
  if (receipt.get('status') != 'pass' or receipt.get('gpu_only') is not True or
      receipt.get('fixture_manifest_sha256') != digest(fixtures/'manifest.json') or
      receipt.get('device_output_sha256') != digest(device_output/'actual.f32') or
      receipt.get('remote_directory_removed') is not True):
    raise ValueError('GPU result or fixture provenance mismatch')
  refs = ROOT/'android/app/build/model-reference'
  expected_model = json.loads((refs/'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  original = ROOT/'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  if manifest.get('original_sha256') != expected_model or digest(original) != expected_model:
    raise ValueError('Original road model provenance mismatch')
  models = {}
  for name in ('prefix', 'dense', 'suffix'):
    path = sections_dir/(name+'.onnx')
    if digest(path) != manifest['sections'][name]['sha256']:
      raise ValueError('Derived graph hash mismatch')
    models[name] = onnx.load(path)
  dense = models['dense'].graph
  node = dense.node[0]
  weights = {w.name: numpy_helper.to_array(w) for w in dense.initializer}
  for name, value in [('weights', weights[node.input[1]]), ('bias', weights[node.input[2]])]:
    captured = np.fromfile(fixtures/(name+'.f32'), dtype=np.float32)
    if not np.array_equal(captured, value.astype(np.float32).ravel()):
      raise ValueError('GPU fixture does not contain original '+name)
  inputs = initial_inputs(refs, 'driving_supercombo', seed)
  prefix = Recompose([('prefix', models['prefix'])], manifest['prefix_crossings'])(inputs)
  captured_input = np.fromfile(fixtures/'input.f32', dtype=np.float32)
  if not np.array_equal(prefix['model/view'].astype(np.float32).ravel(), captured_input):
    raise ValueError('Device dense input differs from recomposed original prefix')
  gpu = np.fromfile(device_output/'actual.f32', dtype=np.float32)
  if gpu.shape != (2048,) or not np.isfinite(gpu).all() or not np.array_equal(gpu, gpu.astype(np.float16).astype(np.float32)):
    raise ValueError('Invalid GPU dense output shape or precision')
  suffix_inputs = {**inputs, **prefix, 'model/linear': gpu.astype(np.float16).reshape(1, -1)}
  suffix_outputs = [v.name for v in models['suffix'].graph.output]
  suffix = Recompose([('suffix', models['suffix'])], suffix_outputs)(suffix_inputs)
  combined = {**prefix, **suffix}
  reference_name = 'driving_supercombo' + ('' if seed == 42 else f'-case-{seed}') + '-outputs.npz'
  with np.load(refs/reference_name) as reference:
    if set(manifest['public_outputs']) != set(reference.files):
      raise ValueError('Original public outputs changed')
    results = {name: compare(combined[name], reference[name]) for name in manifest['public_outputs']}
  return {'seed': seed, 'passed': all(v['passed'] for v in results.values()), 'outputs': results,
          'device_output_sha256': receipt['device_output_sha256'],
          'fixture_manifest_sha256': receipt['fixture_manifest_sha256'],
          'probe_files': receipt['files'], 'prefix_input_exact': True}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--sections', type=Path, required=True)
  parser.add_argument('--case', nargs=3, action='append', metavar=('SEED', 'FIXTURES', 'DEVICE_OUTPUT'), required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise ValueError('Preserve earlier recomposition evidence')
  results = [validate_case(args.sections, Path(fixture), Path(output), int(seed))
             for seed, fixture, output in args.case]
  report = {'passed': all(x['passed'] for x in results), 'diagnostic_only': True, 'qualified': False,
            'vehicle_tx_enabled': False, 'scope': 'host prefix and suffix with saved Fold 3 GPU dense output',
            'not_measured': ['full device graph', 'recurrent sequences', 'replay', 'end-to-end timing'],
            'sections_manifest_sha256': digest(args.sections/'manifest.json'), 'cases': results}
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'passed': report['passed'], 'seeds': [x['seed'] for x in results]}))
  return 0 if report['passed'] else 1


if __name__ == '__main__':
  raise SystemExit(main())
