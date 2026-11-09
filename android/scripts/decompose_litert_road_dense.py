#!/usr/bin/env python3
"""Split an already reference-validated LiteRT road rewrite around dense."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx

from decompose_road import ROOT, digest, reference_checks
from decompose_road_dense import decompose, OrderedDenseRecompose


def adapted_runner(runner):
  def run(inputs):
    values = dict(inputs)
    values['new_img'] = values['new_img'].astype(np.float32)
    values['state_img_q'] = values['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
    out = runner(values)
    history = out['next_state_img_q'].reshape(2, 5, 6, 128, 256)
    if not np.isfinite(history).all() or not ((history >= 0) & (history <= 255) & (history == np.floor(history))).all():
      raise ValueError('Packed image history changed byte values')
    out['next_state_img_q'] = history.astype(np.uint8)
    return out
  return run


def prepare(source, receipt_path, output):
  refs = ROOT/'android/app/build/model-reference'
  original = ROOT/'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  expected = json.loads((refs/'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  receipt = json.loads(receipt_path.read_text())
  if (digest(original) != expected or receipt['original_sha256'] != expected or
      receipt['derived_sha256'] != digest(source) or not receipt['host_reference']['passed'] or
      receipt['host_reference']['comparisons'] != 636):
    raise ValueError('Input graph has not passed original road references')
  sections, manifest = decompose(onnx.load(source), expected_dimensions=(2048, 1024))
  output.mkdir(parents=True, exist_ok=False)
  manifest.update(original_sha256=expected, rewritten_source_sha256=digest(source),
                  source_receipt_sha256=digest(receipt_path), diagnostic_only=True,
                  qualified=False, vehicle_tx_enabled=False, rtol=1e-3, atol=1e-4, sections={})
  for name, graph in sections:
    path = output/(name+'.onnx')
    onnx.save(graph, path)
    manifest['sections'][name] = {'sha256': digest(path), 'nodes': len(graph.graph.node),
                                  'inputs': [v.name for v in graph.graph.input],
                                  'outputs': [v.name for v in graph.graph.output]}
  (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
  runner = OrderedDenseRecompose(sections, manifest['public_outputs'])
  report = reference_checks(adapted_runner(runner), refs, 'driving_supercombo')
  report.update(diagnostic_only=True, qualified=False, vehicle_tx_enabled=False,
                provider='host CPU only with ordered-half dense',
                rewritten_source_sha256=digest(source), sections_manifest_sha256=digest(output/'manifest.json'))
  (output/'host-ordered-recomposition.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps({'passed': report['passed'], 'comparisons': report['comparisons']}), flush=True)
  return report['passed'] and report['comparisons'] == 636


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for arg in ('source', 'receipt', 'output'):
    parser.add_argument('--'+arg, type=Path, required=True)
  args = parser.parse_args()
  raise SystemExit(0 if prepare(args.source, args.receipt, args.output) else 1)
