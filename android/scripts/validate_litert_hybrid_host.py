#!/usr/bin/env python3
"""Run the converted sections together on CPU; diagnostic, never qualifies GPU."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from onnx import numpy_helper

from decompose_road import ROOT, digest
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from prepare_snpe_dense_probe import ordered_half_gemv
from validate_litert_dense_sections_host import run_converted, run_onnx, session


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('sections', 'prefix', 'suffix', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise ValueError('Preserve earlier evidence')
  manifest = json.loads((args.sections / 'manifest.json').read_text())
  refs = ROOT / 'android/app/build/model-reference'
  expected = json.loads((refs / 'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  original = ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  if manifest['original_sha256'] != expected or digest(original) != expected:
    raise ValueError('Original model hash mismatch')
  for name in ('prefix', 'dense', 'suffix'):
    if digest(args.sections / (name + '.onnx')) != manifest['sections'][name]['sha256']:
      raise ValueError('Section hash mismatch: ' + name)
  graph = onnx.load(args.sections / 'dense.onnx').graph
  if len(graph.node) != 1 or graph.node[0].op_type != 'Gemm':
    raise ValueError('Dense section contract changed')
  weights = {w.name: numpy_helper.to_array(w) for w in graph.initializer}
  weight, bias = (weights[name] for name in graph.node[0].input[1:])
  prefix_onnx, suffix_onnx = (session(args.sections / (n + '.onnx')) for n in ('prefix', 'suffix'))
  converted = {}
  for name in ('prefix', 'suffix'):
    model = Interpreter(model_path=str(getattr(args, name)), num_threads=2,
                        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    model.allocate_tensors()
    converted[name] = model
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'provider': 'host LiteRT CPU and ordered-half CPU diagnostic',
            'original_sha256': expected, 'sections_manifest_sha256': digest(args.sections / 'manifest.json'),
            'converted_sha256': {n: digest(getattr(args, n)) for n in ('prefix', 'suffix')}, 'seeds': {}}
  for seed in (42, 7, 123):
    inputs = initial_inputs(refs, 'driving_supercombo', seed)
    if 'rewritten_source_sha256' in manifest:
      inputs['new_img'] = inputs['new_img'].astype(np.float32)
      inputs['state_img_q'] = inputs['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
    prefix_inputs = {v.name: inputs[v.name] for v in prefix_onnx.get_inputs()}
    prefix = run_converted(converted['prefix'], prefix_inputs)
    reference_prefix = run_onnx(prefix_onnx, inputs)
    linear = ordered_half_gemv(prefix['model/view'], weight, bias)
    reference_linear = ordered_half_gemv(reference_prefix['model/view'], weight, bias)
    suffix_inputs = {**inputs, **prefix, 'model/linear': linear}
    reference_suffix_inputs = {**inputs, **reference_prefix, 'model/linear': reference_linear}
    inherited_suffix = run_onnx(suffix_onnx, suffix_inputs)
    reference_suffix = run_onnx(suffix_onnx, reference_suffix_inputs)
    selected = {v.name: suffix_inputs[v.name] for v in suffix_onnx.get_inputs()}
    suffix = run_converted(converted['suffix'], selected)
    actual = {**prefix, **suffix}
    reference_name = 'driving_supercombo' + ('' if seed == 42 else f'-case-{seed}') + '-outputs.npz'
    with np.load(refs / reference_name) as reference:
      if set(manifest['public_outputs']) != set(reference.files):
        raise ValueError('Reference outputs changed')
      outputs = {}
      for name in manifest['public_outputs']:
        value = actual[name]
        if name == 'next_state_img_q' and value.size == reference[name].size:
          value = value.reshape(reference[name].shape)
        if value.shape != reference[name].shape:
          raise ValueError('Output shape mismatch: ' + name)
        outputs[name] = compare(value, reference[name])
    inherited = {name: compare(value, reference_suffix[name]) for name, value in inherited_suffix.items()}
    report['seeds'][str(seed)] = {'prefix_view': compare(prefix['model/view'], reference_prefix['model/view']),
                                 'dense_output': compare(linear, reference_linear),
                                 'onnx_suffix_from_converted_prefix': inherited, 'outputs': outputs}
  report['passed'] = all(x['passed'] for case in report['seeds'].values() for x in case['outputs'].values())
  report['not_measured'] = ['GPU execution', 'recurrent sequences', 'replay', 'latency']
  args.output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps({'passed': report['passed'], 'seeds': report['seeds']}))
  return 0 if report['passed'] else 1


if __name__ == '__main__':
  raise SystemExit(main())
