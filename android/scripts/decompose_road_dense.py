#!/usr/bin/env python3
"""Check a diagnostic road split around the original ordered-half dense layer."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from decompose_road import ROOT, digest, reference_checks, Recompose, split
from prepare_snpe_dense_probe import ordered_half_gemv


def decompose(model, expected_dimensions=None):
  first, outer = split(model, 'model/view')
  second, inner = split(first[1][1], 'model/linear')
  dense = second[0][1]
  nodes = list(dense.graph.node)
  if len(nodes) != 1 or nodes[0].op_type != 'Gemm':
    raise ValueError('Dense boundary is not a single Gemm')
  node = nodes[0]
  attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
  if (list(node.output) != ['model/linear'] or len(node.input) != 3 or
      node.input[0] != 'model/view' or
      attributes != {'alpha': 1., 'beta': 1., 'transA': 0, 'transB': 1}):
    raise ValueError('Dense arithmetic contract changed')
  weights = {v.name: v for v in dense.graph.initializer}
  if any(name not in weights for name in node.input[1:]):
    raise ValueError('Dense constants missing')
  weight, bias = (numpy_helper.to_array(weights[name]) for name in node.input[1:])
  if (weight.dtype != np.float16 or bias.dtype != np.float16 or weight.ndim != 2 or
      weight.shape[1] % 8 or bias.shape != (weight.shape[0],)):
    raise ValueError('Dense weight, bias or half-precision contract changed')
  if expected_dimensions and weight.shape != expected_dimensions:
    raise ValueError('Dense dimensions changed')
  for name, contract in [('model/view', outer), ('model/linear', inner)]:
    if (name not in contract['crossing_tensors'] or
        contract['tensor_contracts'][name]['onnx_dtype'] != TensorProto.FLOAT16):
      raise ValueError('Dense boundary is not a crossing half tensor')
  if inner['crossing_tensors'] != ['model/linear']:
    raise ValueError('Dense operation has unexpected crossing outputs')
  sections = [('prefix', first[0][1]), ('dense', dense), ('suffix', second[1][1])]
  return sections, {'public_outputs': outer['public_outputs'],
                    'prefix_crossings': outer['crossing_tensors'],
                    'dense_crossings': inner['crossing_tensors'],
                    'boundary_contracts': {k: outer['tensor_contracts'][k]
                                           for k in outer['crossing_tensors']},
                    'dense_output_contract': inner['tensor_contracts']['model/linear'],
                    'dense_node': node.name, 'dense_attributes': attributes,
                    'weight_shape': list(weight.shape), 'bias_shape': list(bias.shape)}


class OrderedDenseRecompose:
  """Host diagnostic only; the device kernel is not integrated into this runner."""
  def __init__(self, sections, outputs):
    self.outputs, self.sessions, self.timings = outputs, {}, []
    dense = dict(sections)['dense'].graph
    node = dense.node[0]
    weights = {v.name: numpy_helper.to_array(v) for v in dense.initializer}
    self.weight, self.bias = (weights[name] for name in node.input[1:])
    for name, model in sections:
      if name == 'dense':
        continue
      options = ort.SessionOptions()
      options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
      self.sessions[name] = ort.InferenceSession(model.SerializeToString(), options,
                                                  providers=['CPUExecutionProvider'])

  def __call__(self, inputs):
    values = dict(inputs)
    for name in ('prefix', 'dense', 'suffix'):
      start = time.perf_counter_ns()
      if name == 'dense':
        values['model/linear'] = ordered_half_gemv(values['model/view'], self.weight, self.bias)
      else:
        session = self.sessions[name]
        outputs = session.run(None, {v.name: values[v.name] for v in session.get_inputs()})
        values.update(zip((v.name for v in session.get_outputs()), outputs, strict=True))
      self.timings.append({'section': name, 'wall_ms': (time.perf_counter_ns()-start)/1e6})
    return {name: values[name] for name in self.outputs}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--seeds-only', action='store_true')
  parser.add_argument('--include-driver', action='store_true')
  args = parser.parse_args()
  original = ROOT/'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  references = ROOT/'android/app/build/model-reference'
  expected = json.loads((references/'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if digest(original) != expected:
    raise ValueError('Original road model hash mismatch')
  sections, manifest = decompose(onnx.load(original), expected_dimensions=(2048, 1024))
  args.output.mkdir(parents=True, exist_ok=False)
  manifest.update(original_sha256=expected, diagnostic_only=True, qualified=False,
                  vehicle_tx_enabled=False, rtol=1e-3, atol=1e-4, sections={})
  for name, graph in sections:
    path = args.output/(name+'.onnx')
    onnx.save(graph, path)
    manifest['sections'][name] = {'sha256': digest(path), 'nodes': len(graph.graph.node),
                                  'inputs': [v.name for v in graph.graph.input],
                                  'outputs': [v.name for v in graph.graph.output]}
  (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
  runner = OrderedDenseRecompose(sections, manifest['public_outputs'])
  report = reference_checks(runner, references, 'driving_supercombo', args.seeds_only)
  report.update(original_sha256=expected, diagnostic_only=True, qualified=False,
                vehicle_tx_enabled=False, provider='host CPU only; NumPy ordered-half dense',
                timing_scope='host section wall times, not GPU or device latency', timings=runner.timings)
  if args.include_driver:
    path = ROOT/'openpilot/selfdrive/modeld/models/dmonitoring_model.onnx'
    expected_driver = json.loads((references/'manifest.json').read_text())['models']['dmonitoring_model']['sha256']
    if digest(path) != expected_driver:
      raise ValueError('Original driver model hash mismatch')
    model = onnx.load(path)
    driver = Recompose([('driver', model)], [v.name for v in model.graph.output])
    report['driver'] = reference_checks(driver, references, 'dmonitoring_model', args.seeds_only)
    report['driver']['original_sha256'] = expected_driver
    report['passed'] = report['passed'] and report['driver']['passed']
    report['total_comparisons'] = report['comparisons'] + report['driver']['comparisons']
  (args.output/'host-ordered-recomposition.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'passed': report['passed'], 'road_comparisons': report['comparisons'],
                    'total_comparisons': report.get('total_comparisons'),
                    'seeds_only': args.seeds_only}), flush=True)
  return 0 if report['passed'] else 1


if __name__ == '__main__':
  raise SystemExit(main())
