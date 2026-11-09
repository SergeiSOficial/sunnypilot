#!/usr/bin/env python3
"""Trace suffix tensors one at a time and check the untapped public outputs."""
import argparse
import copy
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
from trace_litert_prefix import tap_impact
from validate_litert_dense_sections_host import run_converted, run_onnx, session


def compare_axis(actual, expected, axis):
  if actual.shape != expected.shape or not 0 <= axis < actual.ndim or not 0 < actual.shape[axis] <= 256:
    raise ValueError('Invalid comparison axis or tensor shape')
  return [{'index': index, **compare(np.take(actual, index, axis=axis), np.take(expected, index, axis=axis))}
          for index in range(actual.shape[axis])]


def trace(sections, conversion, model, output, names, seed, derived_sha256=None, axis=None):
  if output.exists() or not names or len(set(names)) != len(names):
    raise ValueError('Use a new report and unique tensor names')
  manifest = json.loads((sections / 'manifest.json').read_text())
  receipt = json.loads((conversion / 'receipt.json').read_text())
  refs = ROOT / 'android/app/build/model-reference'
  expected_hash = json.loads((refs / 'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if digest(ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx') != expected_hash or manifest['original_sha256'] != expected_hash:
    raise ValueError('Original model provenance mismatch')
  for name in ('prefix', 'dense', 'suffix'):
    if digest(sections / (name + '.onnx')) != manifest['sections'][name]['sha256']:
      raise ValueError('Section provenance mismatch')
  if digest(sections / 'suffix.onnx') != receipt['source_sha256']:
    raise ValueError('Conversion provenance mismatch')
  actual_hash = digest(model)
  if derived_sha256:
    if actual_hash != derived_sha256:
      raise ValueError('Derived model hash mismatch')
  elif actual_hash != receipt['outputs'].get(str(model.relative_to(conversion))):
    raise ValueError('Conversion provenance mismatch')
  inputs = initial_inputs(refs, 'driving_supercombo', seed)
  if 'rewritten_source_sha256' in manifest:
    inputs['new_img'] = inputs['new_img'].astype(np.float32)
    inputs['state_img_q'] = inputs['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
  prefix = run_onnx(session(sections / 'prefix.onnx'), inputs)
  dense = onnx.load(sections / 'dense.onnx').graph
  weights = {v.name: numpy_helper.to_array(v) for v in dense.initializer}
  inputs.update(prefix)
  inputs['model/linear'] = ordered_half_gemv(prefix['model/view'], *(weights[n] for n in dense.node[0].input[1:]))
  untouched = run_onnx(session(sections / 'suffix.onnx'), inputs)
  graph = onnx.shape_inference.infer_shapes(onnx.load(sections / 'suffix.onnx'))
  infos = {v.name: v for v in (*graph.graph.value_info, *graph.graph.output)}
  nodes = {name: n for n in graph.graph.node for name in n.output}
  mappings = [name.split('=', 1) if '=' in name else [name, name] for name in names]
  onnx_names = [item[0] for item in mappings]
  if not set(onnx_names) <= infos.keys() or not set(onnx_names) <= nodes.keys():
    raise ValueError('Requested ONNX tensor missing')
  runner = Interpreter(model_path=str(model), num_threads=2, experimental_preserve_all_tensors=True,
                       experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
  runner.allocate_tensors()
  run_converted(runner, {d['name']: inputs[d['name']] for d in runner.get_input_details()})
  details = {d['name']: d for d in runner.get_tensor_details()}
  operations = runner._get_ops_details()
  options = ort.SessionOptions()
  options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
  records = []
  for name, litert_name in mappings:
    if litert_name not in details:
      raise ValueError('Requested converted tensor missing: ' + litert_name)
    graph.graph.output.append(copy.deepcopy(infos[name]))
    serialized = graph.SerializeToString()
    graph.graph.output.pop()
    tapped = run_onnx(ort.InferenceSession(serialized, options, providers=['CPUExecutionProvider']), inputs)
    impact = tap_impact(untouched, tapped)
    actual = runner.get_tensor(details[litert_name]['index'])
    producers = [op for op in operations if details[litert_name]['index'] in op['outputs']]
    if len(producers) != 1:
      raise ValueError('Expected a unique converted tensor producer')
    record = {'tensor': name, 'litert_tensor': litert_name, 'onnx_node': nodes[name].name, 'onnx_operator': nodes[name].op_type,
              'litert_operator': producers[0]['op_name'], 'litert_operator_index': producers[0]['index'],
              'comparison': compare(actual, tapped[name]), 'tap_impact': impact}
    if axis is not None:
      record['axis'] = axis
      record['axis_comparisons'] = compare_axis(actual, tapped[name], axis)
    records.append(record)
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'provider': 'host LiteRT CPU and individually tapped ONNX CPU; not GPU evidence',
            'warning': 'Preserving public outputs does not prove intermediate execution is unchanged',
            'seed': seed, 'original_sha256': expected_hash, 'source_sha256': digest(sections / 'suffix.onnx'),
            'model_sha256': digest(model), 'records': records}
  output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps([{'tensor': r['tensor'], 'mismatches': r['comparison']['mismatched_elements'],
                     'tap_output_tolerance_preserved': r['tap_impact']['tolerance_preserved']} for r in records], indent=2))


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('sections', 'conversion', 'model', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
  p.add_argument('--tensor', action='append', required=True)
  p.add_argument('--derived-sha256', help='Hash of a diagnostic derived model; not conversion qualification')
  p.add_argument('--axis', type=int, help='Also report comparisons per index of this tensor axis')
  p.add_argument('--seed', type=int, default=42)
  args = p.parse_args()
  trace(args.sections, args.conversion, args.model, args.output, args.tensor, args.seed, args.derived_sha256, args.axis)
