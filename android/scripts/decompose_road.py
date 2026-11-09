#!/usr/bin/env python3
"""Exact dependency-cut road graphs; original weights and arithmetic are untouched."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'android/app/src/main/python'))
from fold3_model_trace import compare
from fold3_sequences import initial_inputs, validate


def digest(path):
  with Path(path).open('rb') as f:
    return hashlib.file_digest(f, 'sha256').hexdigest()


def split(model, boundary='model/vision_outputs'):
  """Cut the entire ancestor closure, including skip connections and final outputs."""
  model = onnx.shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
  producers = {v: i for i, n in enumerate(model.graph.node) for v in n.output if v}
  if boundary not in producers:
    raise ValueError('Missing road vision boundary')
  before = set()
  def visit(value):
    i = producers.get(value)
    if i is not None and i not in before:
      before.add(i)
      for v in model.graph.node[i].input:
        visit(v)
  visit(boundary)
  public = [v.name for v in model.graph.output]
  produced = {v for i in before for v in model.graph.node[i].output}
  needed = {v for i, n in enumerate(model.graph.node) if i not in before for v in n.input}
  crossing = sorted(produced & (needed | set(public)))
  types = {v.name: v for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  weights = {v.name: v for v in model.graph.initializer}
  sections = []
  for name, indices, outputs in [('vision', before, crossing),
                                 ('policy', set(range(len(model.graph.node))) - before, [v for v in public if v not in produced])]:
    nodes = [n for i, n in enumerate(model.graph.node) if i in indices]
    generated = {v for n in nodes for v in n.output}
    external = {v for n in nodes for v in n.input if v and v not in generated}
    inputs = sorted(external - weights.keys())
    if any(v not in types for v in inputs + outputs):
      raise ValueError('Unknown crossing tensor contract')
    graph = helper.make_graph(nodes, name, [types[v] for v in inputs], [types[v] for v in outputs],
                              [weights[v] for v in sorted(external & weights.keys())])
    out = copy.deepcopy(model)
    out.graph.CopyFrom(graph)
    onnx.checker.check_model(out)
    sections.append((name, out))
  return sections, {'boundary': boundary, 'crossing_tensors': crossing, 'public_outputs': public,
                    'tensor_contracts': {name: {'onnx_dtype': v.type.tensor_type.elem_type,
                      'shape': [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim]}
                      for name, v in types.items()},
                    'layout': 'original ONNX axes; no transpose or precision rewrite',
                    'operator_counts': {name: dict(Counter(n.op_type for n in graph.graph.node))
                                        for name, graph in sections},
                    'nodes': [{'index': i, 'name': n.name, 'operator': n.op_type,
                               'section': 'vision' if i in before else 'policy',
                               'inputs': list(n.input), 'outputs': list(n.output)}
                              for i, n in enumerate(model.graph.node)]}


class Recompose:
  def __init__(self, sections, outputs):
    self.outputs, self.sessions, self.timings = outputs, [], []
    for name, model in sections:
      options = ort.SessionOptions()
      options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
      start = time.perf_counter_ns()
      session = ort.InferenceSession(model.SerializeToString(), options, providers=['CPUExecutionProvider'])
      self.sessions.append((name, session))
      self.timings.append({'section': name, 'initialization_ms': (time.perf_counter_ns()-start)/1e6})

  def __call__(self, inputs):
    values = dict(inputs)
    for name, session in self.sessions:
      start = time.perf_counter_ns()
      outputs = session.run(None, {v.name: values[v.name] for v in session.get_inputs()})
      values.update(zip((v.name for v in session.get_outputs()), outputs, strict=True))
      self.timings.append({'section': name, 'wall_ms': (time.perf_counter_ns()-start)/1e6})
    return {name: values[name] for name in self.outputs}


def reference_checks(runner, references, name, seeds_only=False):
  report = {'initial': {}}
  for seed in (42, 7, 123):
    actual = runner(initial_inputs(references, name, seed))
    suffix = '' if seed == 42 else f'-case-{seed}'
    with np.load(references/f'{name}{suffix}-outputs.npz') as expected:
      if set(actual) != set(expected.files): raise ValueError('Original output contract changed')
      report['initial'][str(seed)] = {k: compare(v, expected[k]) for k, v in actual.items()}
  if not seeds_only: report['sequences'] = validate(runner, references, name)
  report['passed'] = all(v['passed'] for seed in report['initial'].values() for v in seed.values()) and all(
    v['passed'] for v in report.get('sequences', {}).values())
  report['comparisons'] = sum(len(seed) for seed in report['initial'].values()) + sum(
    len(check['outputs']) for sequence in report.get('sequences', {}).values() for check in sequence['checks'])
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--seeds-only', action='store_true')
  parser.add_argument('--include-driver', action='store_true', help='Also run untouched host driver regression (159 comparisons)')
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  original = ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  references = ROOT / 'android/app/build/model-reference'
  expected_hash = json.loads((references/'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if digest(original) != expected_hash:
    raise ValueError('Original model hash mismatch')
  sections, manifest = split(onnx.load(original))
  manifest.update(original_sha256=expected_hash, diagnostic_only=True, qualified=False,
                  rtol=1e-3, atol=1e-4, vehicle_tx_enabled=False, sections={})
  for name, graph in sections:
    path = args.output / (name+'.onnx'); onnx.save(graph, path)
    manifest['sections'][name] = {'sha256': digest(path), 'nodes': len(graph.graph.node),
      'inputs': {v.name: {'shape': [d.dim_value or d.dim_param for d in v.type.tensor_type.shape.dim],
                         'onnx_dtype': v.type.tensor_type.elem_type} for v in graph.graph.input},
      'outputs': [v.name for v in graph.graph.output]}
  (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
  report = {'original_sha256': expected_hash, 'diagnostic_only': True, 'qualified': False,
            'vehicle_tx_enabled': False, 'provider': 'host CPU reference only', 'initial': {}}
  runner = Recompose(sections, manifest['public_outputs'])
  report.update(reference_checks(runner, references, 'driving_supercombo', args.seeds_only))
  report['timings'] = runner.timings
  report['timing_scope'] = 'host CPU section wall times, including ORT copies; not GPU or device latency'
  if args.include_driver:
    path = ROOT/'openpilot/selfdrive/modeld/models/dmonitoring_model.onnx'
    expected = json.loads((references/'manifest.json').read_text())['models']['dmonitoring_model']['sha256']
    if digest(path) != expected: raise ValueError('Original driver model hash mismatch')
    model = onnx.load(path)
    driver = Recompose([('driver', model)], [v.name for v in model.graph.output])
    report['driver'] = reference_checks(driver, references, 'dmonitoring_model', args.seeds_only)
    report['driver']['original_sha256'] = expected
    report['passed'] = report['passed'] and report['driver']['passed']
    report['total_comparisons'] = report['comparisons'] + report['driver']['comparisons']
  (args.output/'host-recomposition.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'passed': report['passed'], 'sequence_validation': not args.seeds_only,
                    'sections': {k: v['nodes'] for k,v in manifest['sections'].items()}}), flush=True)
  return 0 if report['passed'] else 1


if __name__ == '__main__':
  raise SystemExit(main())
