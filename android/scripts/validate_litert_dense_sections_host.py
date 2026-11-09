#!/usr/bin/env python3
"""Compare converted road sections against pinned ONNX; CPU diagnostic only."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from onnx import numpy_helper

from decompose_road import ROOT, digest
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from prepare_snpe_dense_probe import ordered_half_gemv


def convert_input(name, value, detail):
  # These axes are specific to the recorded conversion; reject any changed contract.
  axes = {'new_img': (0, 2, 3, 1), 'state_img_q': (0, 2, 3, 4, 1),
          'state_desire_q': (0, 2, 1), 'state_feat_q': (0, 2, 1)}
  value = np.asarray(value)
  if name == 'state_img_q' and value.ndim == 4:
    axes[name] = (0, 2, 3, 1)
  if name in axes:
    value = value.transpose(axes[name])
  value = np.ascontiguousarray(value, dtype=detail['dtype'])
  if tuple(value.shape) != tuple(detail['shape']):
    raise ValueError(f'Unexpected converted input shape for {name}: {value.shape} != {detail["shape"]}')
  return value


def run_converted(model, inputs):
  details = model.get_input_details()
  if set(inputs) != {d['name'] for d in details}:
    raise ValueError('Converted input names differ from ONNX section')
  for detail in details:
    model.set_tensor(detail['index'], convert_input(detail['name'], inputs[detail['name']], detail))
  model.invoke()
  return {d['name']: model.get_tensor(d['index']) for d in model.get_output_details()}


def session(path):
  options = ort.SessionOptions()
  options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
  return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def run_onnx(s, values):
  outputs = s.run(None, {v.name: values[v.name] for v in s.get_inputs()})
  return dict(zip((v.name for v in s.get_outputs()), outputs, strict=True))


def check_outputs(actual, expected):
  if set(actual) != set(expected):
    raise ValueError('Converted output names differ from ONNX section')
  result = {}
  for name, want in expected.items():
    got = actual[name]
    if got.shape != want.shape:
      raise ValueError(f'Unexpected output shape for {name}: {got.shape} != {want.shape}')
    result[name] = compare(got, want)
  return result


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--sections', type=Path, required=True)
  p.add_argument('--prefix', type=Path, required=True)
  p.add_argument('--suffix', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  if args.output.exists():
    raise ValueError('Preserve prior evidence')
  manifest = json.loads((args.sections/'manifest.json').read_text())
  references = ROOT/'android/app/build/model-reference'
  expected_hash = json.loads((references/'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if manifest['original_sha256'] != expected_hash or digest(ROOT/'openpilot/selfdrive/modeld/models/driving_supercombo.onnx') != expected_hash:
    raise ValueError('Original model hash mismatch')
  for name in ('prefix', 'dense', 'suffix'):
    if digest(args.sections/(name+'.onnx')) != manifest['sections'][name]['sha256']:
      raise ValueError('Section hash mismatch: '+name)
  onnx_sessions = {name: session(args.sections/(name+'.onnx')) for name in ('prefix', 'suffix')}
  graph = onnx.load(args.sections/'dense.onnx').graph
  weights = {w.name: numpy_helper.to_array(w) for w in graph.initializer}
  node = graph.node[0]
  weight, bias = (weights[name] for name in node.input[1:])
  converted = {}
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'provider': 'host LiteRT CPU only, no default delegates', 'original_sha256': expected_hash,
            'sections_manifest_sha256': digest(args.sections/'manifest.json'),
            'converted_sha256': {n: digest(getattr(args, n)) for n in ('prefix', 'suffix')},
            'seeds': {}, 'passed': False}
  for name in ('prefix', 'suffix'):
    try:
      converted[name] = Interpreter(model_path=str(getattr(args, name)), num_threads=2,
                                    experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
      converted[name].allocate_tensors()
    except Exception as exc:
      report.setdefault('initialization_errors', {})[name] = f'{type(exc).__name__}: {exc}'
  for seed in (42, 7, 123):
    case = report['seeds'][str(seed)] = {}
    inputs = initial_inputs(references, 'driving_supercombo', seed)
    if 'rewritten_source_sha256' in manifest:
      inputs['new_img'] = inputs['new_img'].astype(np.float32)
      inputs['state_img_q'] = inputs['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
    original_prefix = run_onnx(onnx_sessions['prefix'], inputs)
    linear = ordered_half_gemv(original_prefix['model/view'], weight, bias)
    original_suffix_inputs = {**inputs, **original_prefix, 'model/linear': linear}
    original_suffix = run_onnx(onnx_sessions['suffix'], original_suffix_inputs)
    for name, expected in [('prefix', original_prefix), ('suffix', original_suffix)]:
      try:
        if name not in converted:
          raise RuntimeError('Interpreter initialization failed')
        source = inputs if name == 'prefix' else original_suffix_inputs
        selected = {v.name: source[v.name] for v in onnx_sessions[name].get_inputs()}
        actual = run_converted(converted[name], selected)
        checks = check_outputs(actual, expected)
        case[name] = {'passed': all(v['passed'] for v in checks.values()), 'outputs': checks}
      except Exception as exc:
        case[name] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
  report['passed'] = all(case[name]['passed'] for case in report['seeds'].values() for name in ('prefix', 'suffix'))
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'passed': report['passed'], 'seeds': report['seeds']}, indent=2))
  return 0 if report['passed'] else 1


if __name__ == '__main__':
  raise SystemExit(main())
