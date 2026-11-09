#!/usr/bin/env python3
"""Trace matching ONNX/LiteRT suffix tensors; diagnostic CPU execution only."""
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
from trace_litert_prefix import tap_impact
from validate_litert_dense_sections_host import run_converted, run_onnx, session


def trace(sections, model, correspondence, output, seed):
  if output.exists():
    raise ValueError('Use a new output directory')
  manifest = json.loads((sections / 'manifest.json').read_text())
  for name in ('prefix', 'dense', 'suffix'):
    if digest(sections / (name + '.onnx')) != manifest['sections'][name]['sha256']:
      raise ValueError('Section hash mismatch: ' + name)
  output.mkdir(parents=True)
  reference = ROOT / 'android/app/build/model-reference'
  inputs = initial_inputs(reference, 'driving_supercombo', seed)
  inputs['new_img'] = inputs['new_img'].astype(np.float32)
  inputs['state_img_q'] = inputs['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
  prefix = run_onnx(session(sections / 'prefix.onnx'), inputs)
  dense = onnx.load(sections / 'dense.onnx').graph
  weights = {v.name: numpy_helper.to_array(v) for v in dense.initializer}
  linear = ordered_half_gemv(prefix['model/view'], *(weights[n] for n in dense.node[0].input[1:]))
  values = {**inputs, **prefix, 'model/linear': linear}
  graph = onnx.shape_inference.infer_shapes(onnx.load(sections / 'suffix.onnx'))
  inferred = {v.name: v for v in (*graph.graph.value_info, *graph.graph.output)}
  existing = {v.name for v in graph.graph.output}
  records = json.loads(correspondence.read_text())['records']
  positions = {name: index for index, node in enumerate(graph.graph.node) for name in node.output}
  records.sort(key=lambda record: (positions.get(record['onnx_output_name'], len(graph.graph.node)),
                                   record['onnx_output_index']))
  for record in records:
    name = record['onnx_output_name']
    if record['status'] == 'direct' and name in inferred and name not in existing:
      graph.graph.output.append(inferred[name])
      existing.add(name)
  tapped = output / 'suffix-tapped.onnx'
  onnx.save(graph, tapped)
  untouched = run_onnx(session(sections / 'suffix.onnx'), values)
  original = run_onnx(session(tapped), values)
  impact = tap_impact(untouched, original)
  runtime = Interpreter(model_path=str(model), num_threads=2, experimental_preserve_all_tensors=True,
                        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
  runtime.allocate_tensors()
  suffix_inputs = {v.name: values[v.name] for v in session(sections / 'suffix.onnx').get_inputs()}
  run_converted(runtime, suffix_inputs)
  tensors = {v['name']: v for v in runtime.get_tensor_details()}
  operations = runtime._get_ops_details()
  comparisons, skipped, removed_half_boundaries = [], [], []
  for record in records:
    name = record['onnx_output_name']
    detail = tensors.get(record['resolved_tflite_tensor_name'])
    if record['status'] != 'direct':
      continue
    if name not in original or detail is None:
      replacement = tensors.get(name + '_fc_float32')
      if name in original and detail is None and replacement is not None and original[name].dtype == np.float16:
        actual = runtime.get_tensor(replacement['index'])
        if actual.shape == original[name].shape:
          removed_half_boundaries.append({
              'onnx_node': record['onnx_op_name'], 'onnx_tensor': name,
              'litert_tensor': replacement['name'],
              'raw_float32': compare(actual, original[name]),
              'diagnostic_host_half_round': compare(actual.astype(np.float16), original[name])})
      skipped.append({'name': name, 'reason': 'not available in tapped ONNX or final LiteRT graph',
                      'present_in_onnx': name in original, 'present_in_litert': detail is not None})
      continue
    try:
      actual, expected = runtime.get_tensor(detail['index']), original[name]
      producers = [op for op in operations if detail['index'] in op['outputs']]
      if len(producers) != 1:
        raise ValueError('Converted tensor must have one actual producer')
      operation = producers[0]
      if actual.shape != expected.shape:
        skipped.append({'name': name, 'onnx_shape': list(expected.shape), 'litert_shape': list(actual.shape)})
        continue
      result = compare(actual, expected)
      comparisons.append({'onnx_node': record['onnx_op_name'], 'onnx_operator': record['onnx_op_type'],
                          'tensor': name, 'litert_operator': operation['op_name'],
                          'litert_operator_index': operation['index'],
                          'converter_report_operator_index': record['producer_operator_index'],
                          'shape': list(actual.shape), 'onnx_dtype': str(expected.dtype),
                          'litert_dtype': str(actual.dtype), **result})
    except (ValueError, TypeError) as exc:
      skipped.append({'name': name, 'error': str(exc)})
  failures = [item for item in comparisons if not item['passed']]
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'provider': 'host LiteRT CPU and tapped ONNX CPU; not GPU evidence', 'seed': seed,
            'sections_manifest_sha256': digest(sections / 'manifest.json'), 'model_sha256': digest(model),
            'correspondence_sha256': digest(correspondence), 'tapped_sha256': digest(tapped),
            'tap_impact': impact,
            'warning': 'Tapping ONNX outputs may change optimization and fusion.',
            'comparisons': comparisons, 'skipped': skipped, 'removed_half_boundaries': removed_half_boundaries,
            'first_failure': failures[0] if failures else None}
  (output / 'trace.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps({'compared': len(comparisons), 'skipped': len(skipped), 'failures': len(failures),
                    'tap_output_tolerance_preserved': impact['tolerance_preserved'],
                    'first_failure': report['first_failure'], 'removed_half_boundaries': removed_half_boundaries}, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('sections', 'model', 'correspondence', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
  p.add_argument('--seed', type=int, default=42)
  args = p.parse_args()
  trace(args.sections, args.model, args.correspondence, args.output, args.seed)
