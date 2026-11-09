#!/usr/bin/env python3
"""Prepare a hash-bound, diagnostic-only captured road dense fixture."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source, fixtures, output):
  if output.exists():
    raise ValueError('Preserve previous results: output already exists')
  manifest = json.loads((fixtures / 'manifest.json').read_text())
  case = manifest['cases']['dense']
  input_path = fixtures / 'dense' / case['inputs']['input']['file']
  reference_path = fixtures / 'dense' / 'reference.npz'
  if sha(source) != case['section_sha256'] or sha(input_path) != case['inputs']['input']['sha256'] or sha(reference_path) != manifest['reference_sha256']:
    raise ValueError('Fixture provenance does not match the dense manifest')
  graph = onnx.load(source)
  nodes = [node for node in graph.graph.node if node.op_type == 'Gemm']
  if len(nodes) != 1 or nodes[0].name != 'model/node_linear':
    raise ValueError('Unexpected dense graph')
  attrs = {a.name: onnx.helper.get_attribute_value(a) for a in nodes[0].attribute}
  if attrs != {'alpha': 1., 'beta': 1., 'transA': 0, 'transB': 1}:
    raise ValueError('Unexpected dense arithmetic')
  weights = {x.name: numpy_helper.to_array(x) for x in graph.graph.initializer}
  w, b = (weights[name] for name in nodes[0].input[1:])
  x = np.fromfile(input_path, dtype=np.float32)
  with np.load(reference_path) as captured:
    expected = captured['trace_flat'].astype(np.float32).reshape(-1)
  if x.shape != (1024,) or w.shape != (2048, 1024) or b.shape != (2048,) or expected.shape != (2048,):
    raise ValueError('Unexpected dense fixture shape')
  if not all(np.isfinite(v).all() for v in (x, w, b, expected)):
    raise ValueError('Non-finite dense fixture')
  if not np.array_equal(x, x.astype(np.float16).astype(np.float32)):
    raise ValueError('Dense input is not an exact half value')
  output.mkdir(parents=True)
  files = {}
  for name, value in [('input', x), ('weights', w), ('bias', b), ('expected', expected)]:
    path = output / (name + '.f32')
    np.ascontiguousarray(value, dtype=np.float32).tofile(path)
    files[name] = {'file': path.name, 'sha256': sha(path)}
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'graph_sha256': sha(source),
            'manifest_sha256': sha(fixtures / 'manifest.json'), 'inputs': 1024, 'outputs': 2048, 'files': files}
  (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
  return report


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('source', 'fixtures', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
  args = p.parse_args()
  print(json.dumps(prepare(args.source, args.fixtures, args.output), indent=2))
