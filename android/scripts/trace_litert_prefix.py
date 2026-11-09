#!/usr/bin/env python3
"""Trace the road vision prefix on host CPU, without qualifying a backend."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from decompose_road import ROOT, digest
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from validate_litert_dense_sections_host import run_converted, run_onnx, session


def align(actual, expected, channel_last):
  """Align only explicit NHWC or unambiguous shape-different NCHW mappings."""
  if actual.ndim == 4 and expected.ndim == 4:
    transposed = actual.transpose(0, 3, 1, 2)
    if channel_last:
      if transposed.shape != expected.shape:
        raise ValueError('Declared NHWC layout is inconsistent with original shape')
      return transposed, 'NHWC -> NCHW (converter metadata)'
    if actual.shape != expected.shape and transposed.shape == expected.shape:
      return transposed, 'NHWC -> NCHW (unambiguous shape)'
  if actual.shape != expected.shape:
    raise ValueError('Unmapped tensor shape')
  return actual, 'matching shape; layout not independently established'


def producer(operations, tensor_index):
  matches = [operation for operation in operations if tensor_index in operation['outputs']]
  if len(matches) != 1:
    raise ValueError('Converted tensor must have one actual producer')
  return matches[0]


def tap_impact(untapped, tapped):
  """Check public outputs before interpreting a tapped graph's intermediates."""
  if not untapped or not set(untapped).issubset(tapped):
    raise ValueError('Tapped graph is missing original public outputs')
  outputs = {name: compare(tapped[name], value) for name, value in untapped.items()}
  return {'public_outputs': outputs,
          'tolerance_preserved': all(v['passed'] for v in outputs.values()),
          'bitwise_preserved': all(v.get('exact', False) for v in outputs.values())}


def trace(sections, conversion, model, correspondence, output, seed):
  if output.exists():
    raise ValueError('Use a new output directory')
  manifest = json.loads((sections / 'manifest.json').read_text())
  receipt = json.loads((conversion / 'receipt.json').read_text())
  source = sections / 'prefix.onnx'
  if digest(source) != manifest['sections']['prefix']['sha256'] or digest(source) != receipt['source_sha256']:
    raise ValueError('Prefix graph provenance mismatch')
  for path in (model, correspondence):
    relative = str(path.relative_to(conversion))
    if digest(path) != receipt['outputs'].get(relative):
      raise ValueError('Converted artifact provenance mismatch: ' + relative)
  reference = ROOT / 'android/app/build/model-reference'
  original = ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  expected_hash = json.loads((reference / 'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if digest(original) != expected_hash or manifest['original_sha256'] != expected_hash:
    raise ValueError('Original model provenance mismatch')
  inputs = initial_inputs(reference, 'driving_supercombo', seed)
  if 'rewritten_source_sha256' in manifest:
    inputs['new_img'] = inputs['new_img'].astype(np.float32)
    inputs['state_img_q'] = inputs['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
  graph = onnx.shape_inference.infer_shapes(onnx.load(source))
  inferred = {v.name: v for v in (*graph.graph.value_info, *graph.graph.output)}
  existing = {v.name for v in graph.graph.output}
  positions = {name: i for i, node in enumerate(graph.graph.node) for name in node.output}
  nodes = {name: node for node in graph.graph.node for name in node.output}
  records = json.loads(correspondence.read_text())['records']
  selected = [r for r in records if r['status'] == 'direct' and r['onnx_output_name'] in inferred
              and r['onnx_output_name'] in positions and r['producer_operator_index'] is not None]
  selected.sort(key=lambda r: (positions[r['onnx_output_name']], r['onnx_output_index']))
  for record in selected:
    name = record['onnx_output_name']
    for candidate in (name, *nodes[name].input):
      if candidate in inferred and candidate not in existing:
        graph.graph.output.append(inferred[candidate])
        existing.add(candidate)
  output.mkdir(parents=True)
  tapped = output / 'prefix-tapped.onnx'
  onnx.save(graph, tapped)
  untouched_values = run_onnx(session(source), inputs)
  original_values = run_onnx(session(tapped), inputs)
  impact = tap_impact(untouched_values, original_values)
  runtime = Interpreter(model_path=str(model), num_threads=2, experimental_preserve_all_tensors=True,
                        experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
  runtime.allocate_tensors()
  run_converted(runtime, {d['name']: inputs[d['name']] for d in runtime.get_input_details()})
  details = runtime.get_tensor_details()
  tensors = {v['name']: v for v in details}
  tensor_by_index = {v['index']: v for v in details}
  operations = runtime._get_ops_details()
  comparisons, skipped = [], []
  for record in selected:
    name, converted_name = record['onnx_output_name'], record['resolved_tflite_tensor_name']
    detail = tensors.get(converted_name)
    if detail is None:
      skipped.append({'tensor': name, 'reason': 'converted tensor absent'})
      continue
    try:
      raw = runtime.get_tensor(detail['index'])
      operation = producer(operations, detail['index'])
      expected = original_values[name]
      actual, layout = align(raw, expected, record['assume_channel_last_layout'])
      comparisons.append({'onnx_node': record['onnx_op_name'], 'onnx_operator': record['onnx_op_type'],
                          'tensor': name, 'litert_tensor': converted_name, 'litert_operator': operation['op_name'],
                          'litert_operator_index': operation['index'],
                          'converter_report_operator_index': record['producer_operator_index'],
                          'layout': layout, 'onnx_dtype': str(expected.dtype), 'litert_dtype': str(actual.dtype),
                          **compare(actual, expected)})
    except (ValueError, TypeError) as exc:
      skipped.append({'tensor': name, 'reason': str(exc), 'onnx_shape': list(original_values[name].shape),
                      'litert_shape': list(detail['shape'])})
  failures = [item for item in comparisons if not item['passed']]
  isolation = None
  if failures:
    first = failures[0]
    node = nodes[first['tensor']]
    operation = producer(operations, tensors[first['litert_tensor']]['index'])
    if node.op_type in ('Add', 'Sum') and operation['op_name'] == 'ADD' and len(node.input) == len(operation['inputs']) == 2:
      try:
        expected_inputs = [original_values[name] for name in node.input]
        actual_inputs = []
        for index, expected in zip(operation['inputs'], expected_inputs, strict=True):
          value, _ = align(runtime.get_tensor(int(index)), expected, first['layout'].startswith('NHWC'))
          actual_inputs.append(value)
        actual_output, _ = align(runtime.get_tensor(tensors[first['litert_tensor']]['index']),
                                 original_values[first['tensor']], first['layout'].startswith('NHWC'))
        isolation = {'scope': 'host only; positional operands and unfused NumPy addition',
                     'onnx_inputs': list(node.input),
                     'litert_inputs': [tensor_by_index[int(i)]['name'] for i in operation['inputs']],
                     'input_comparisons': [compare(got, want) for got, want in zip(actual_inputs, expected_inputs, strict=True)],
                     'onnx_output_vs_same_input_add': compare(original_values[first['tensor']], np.add(*expected_inputs)),
                     'litert_output_vs_same_input_add': compare(actual_output, np.add(*actual_inputs))}
      except (KeyError, ValueError, TypeError) as exc:
        isolation = {'error': str(exc)}
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False, 'seed': seed,
            'provider': 'host LiteRT CPU and tapped ONNX CPU; not GPU evidence',
            'warning': 'Tapping ONNX tensors may alter graph fusion; matching shapes alone cannot prove layout.',
            'original_sha256': expected_hash, 'sections_manifest_sha256': digest(sections / 'manifest.json'),
            'receipt_sha256': digest(conversion / 'receipt.json'), 'model_sha256': digest(model),
            'correspondence_sha256': digest(correspondence), 'tapped_sha256': digest(tapped),
            'tap_impact': impact,
            'comparisons': comparisons, 'skipped': skipped, 'first_failure': failures[0] if failures else None,
            'first_failure_same_input_add': isolation}
  (output / 'trace.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps({'compared': len(comparisons), 'skipped': len(skipped), 'failures': len(failures),
                    'tap_output_tolerance_preserved': impact['tolerance_preserved'],
                    'first_failure': report['first_failure']}, indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('sections', 'conversion', 'model', 'correspondence', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
  p.add_argument('--seed', type=int, default=42)
  args = p.parse_args()
  trace(args.sections, args.conversion, args.model, args.correspondence, args.output, args.seed)
