#!/usr/bin/env python3
"""Diagnose captured Vulkan tensors without re-fusing the pinned host graph.

Only original tensor names shared by both graphs are compared. Missing layout
aliases are explicit, not inferred from matching shapes. This is never device
qualification; the untapped full-model result remains authoritative.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'android/app/src/main/python'))
sys.path.insert(0, str(Path(__file__).parent))
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from ncnn_replay_inputs import replay_case


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(report):
  manifest = json.loads(report.read_text())
  mapping_path = report.with_name(report.stem + '-comparison.json')
  mapped = json.loads(mapping_path.read_text())
  base = ROOT / 'android/app/build/model-reference'
  portable = base / 'accelerators/dmonitoring_model.onnx'
  if digest(portable) != mapped['original_graph_sha256']:
    raise ValueError('Mapped graph changed')
  weights = base / 'accelerators/ncnn/dmonitoring_model.ncnn.bin'
  if digest(weights) != manifest['sha256']['model.bin']:
    raise ValueError('Mapped weights changed')
  path = base / 'host-fusion/dmonitoring_model.onnx'
  model = onnx.load(path)
  typed = onnx.shape_inference.infer_shapes(onnx.load(portable))
  info = {v.name: v for v in (*typed.graph.input, *typed.graph.value_info, *typed.graph.output)}
  names = {s for n in model.graph.node for s in n.output}
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  inputs = initial_inputs(base, 'dmonitoring_model', manifest['seed'])

  def run(graph):
    return ort.InferenceSession(graph.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, inputs)

  suffix = '' if manifest['seed'] == 42 else '-case-' + str(manifest['seed'])
  golden_path = base / ('dmonitoring_model' + suffix + '-outputs.npz')
  with np.load(golden_path) as saved:
    golden = saved['outputs']
  if manifest.get('replay_frame') is not None:
    inputs, golden = replay_case(base, manifest['replay_frame'])
  original = run(model)[0]
  # Exact preservation here validates the diagnostic graph, not the accelerator.
  if not np.array_equal(original, golden):
    raise ValueError('Frozen host graph does not reproduce the original golden exactly')
  checks, unavailable = [], []
  captures = {t['operator']: t for t in manifest['tensors']}
  for entry in mapped['checks']:
    name = entry['original_tensor']
    if name not in names or name not in info:
      unavailable.append({'operator': entry['operator'], 'tensor': name, 'reason': 'No verified frozen tensor/layout mapping'})
      continue
    tapped = copy.deepcopy(model)
    tapped.graph.output.append(copy.deepcopy(info[name]))
    results = run(tapped)
    if not np.array_equal(results[0], original):
      raise ValueError('Tracing changed the full original output: ' + name)
    tensor = captures[entry['operator']]
    capture = report.with_suffix('') / tensor['file']
    if digest(capture) != tensor['sha256']:
      raise ValueError('Captured tensor changed')
    expected = results[-1]
    if [d for d in tensor['shape'] if d != 1] != [d for d in expected.shape if d != 1]:
      raise ValueError('Captured layout changed: ' + name)
    actual = np.fromfile(capture, np.float32).reshape(expected.shape)
    checks.append({'operator': entry['operator'], 'original_tensor': name,
      'tapped_graph_sha256': hashlib.sha256(tapped.SerializeToString()).hexdigest(),
      'final_output_unchanged': True, 'parity': compare(actual, expected),
      'exact_elements': int(np.count_nonzero(actual == expected)), 'elements': actual.size})
  return {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
    'graph_sha256': digest(path), 'golden_sha256': digest(golden_path), 'trace_sha256': digest(report),
    'mapping_sha256': digest(mapping_path), 'untapped_original_exact': True,
    'accelerator_full_model_parity': manifest['parity'], 'checks': checks, 'unavailable': unavailable,
    'first_compared_tolerance_failure': next((c for c in checks if not c['parity']['passed']), None)}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('report', type=Path)
  args = parser.parse_args()
  result = analyze(args.report)
  output = args.report.with_name(args.report.stem + '-frozen.json')
  output.write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))
