#!/usr/bin/env python3
"""Prepare a diagnostic road graph with proven lower-rank attention layouts."""
import argparse
import json
from pathlib import Path

import onnx
import onnxruntime as ort
import numpy as np

from decompose_road import digest
from prepare_accel_models import check_original, lower_road_history_strides, pack_road_attention


def fold_constant_sqrt(model):
  """Fold only finite, nonnegative float32 initializer square roots."""
  graph = model.graph
  constants = {item.name: onnx.numpy_helper.to_array(item) for item in graph.initializer}
  occupied = {item.name for item in graph.initializer} | {item.name for item in graph.input}
  nodes = []
  count = 0
  for node in graph.node:
    value = constants.get(node.input[0]) if len(node.input) == 1 else None
    if (node.op_type != 'Sqrt' or node.domain or node.attribute or len(node.output) != 1
        or node.output[0] in occupied or value is None or value.dtype != np.float32
        or not np.isfinite(value).all() or not (value >= 0).all()):
      nodes.append(node)
      continue
    graph.initializer.append(onnx.numpy_helper.from_array(np.asarray(np.sqrt(value), dtype=np.float32), node.output[0]))
    occupied.add(node.output[0])
    count += 1
  graph.ClearField('node')
  graph.node.extend(nodes)
  onnx.checker.check_model(model)
  return model, count


def prepare(source, source_sha256, source_receipt, original, references, output, fold_sqrt=False):
  if digest(source) != source_sha256:
    raise ValueError('Source graph hash mismatch')
  manifest = json.loads((references / 'manifest.json').read_text())
  original_sha256 = manifest['models']['driving_supercombo']['sha256']
  prior = json.loads(source_receipt.read_text())
  if (digest(original) != original_sha256 or prior['original_sha256'] != original_sha256
      or prior['derived_sha256'] != source_sha256 or not prior['host_reference']['passed']
      or prior['host_reference']['comparisons'] != 636 or not prior['float_image_transport']):
    raise ValueError('Source provenance or original-reference gate failed')
  output.mkdir(parents=True, exist_ok=False)
  model = lower_road_history_strides(pack_road_attention(onnx.load(source)))
  folded = 0
  if fold_sqrt:
    model, folded = fold_constant_sqrt(model)
  path = output / 'road-low-rank.onnx'
  onnx.save(model, path)
  options = ort.SessionOptions()
  options.intra_op_num_threads = 2
  session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
  reference = check_original(session, references, 'driving_supercombo', packed_history=True,
                             external_outputs={'next_state_img_q': {'type': 'tensor(uint8)'}})
  report = {'qualified': False, 'vehicle_tx_enabled': False,
            'original_sha256': original_sha256, 'source_receipt_sha256': digest(source_receipt),
            'source_sha256': source_sha256, 'derived_sha256': digest(path),
            'rewrites': ['pack attention Q/K/V to rank four', 'lower fixed history strides'], 'constant_sqrt_folded': folded,
            'host_reference': reference}
  (output / 'repair.json').write_text(json.dumps(report, indent=2) + '\n')
  if not reference['passed'] or reference['comparisons'] != 636:
    raise ValueError('Original road reference failed')
  print('PASS', reference['comparisons'], report['derived_sha256'], flush=True)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--source-sha256', required=True)
  parser.add_argument('--source-receipt', type=Path, required=True)
  parser.add_argument('--original', type=Path, required=True)
  parser.add_argument('--reference', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--fold-constant-sqrt', action='store_true')
  args = parser.parse_args()
  prepare(args.source, args.source_sha256, args.source_receipt, args.original, args.reference, args.output, args.fold_constant_sqrt)
