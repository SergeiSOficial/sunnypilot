#!/usr/bin/env python3
"""Compare original ONNX normalization with converted host diagnostic graphs."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert.interpreter import Interpreter, OpResolverType

from decompose_road import ROOT, digest
from fold3_sequences import initial_inputs
from trace_litert_prefix import align
from validate_litert_dense_sections_host import run_converted, run_onnx


def validate(sections, source, modified, output):
  if output.exists():
    raise ValueError('Preserve prior evidence')
  manifest = json.loads((sections / 'manifest.json').read_text())
  prefix = sections / 'prefix.onnx'
  refs = ROOT / 'android/app/build/model-reference'
  original = ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  expected_hash = json.loads((refs / 'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if digest(original) != expected_hash or manifest['original_sha256'] != expected_hash:
    raise ValueError('Original model provenance mismatch')
  if digest(prefix) != manifest['sections']['prefix']['sha256']:
    raise ValueError('Section provenance mismatch')
  graph = onnx.shape_inference.infer_shapes(onnx.load(prefix))
  name = 'InsertedPrecisionFreeCast_model/div'
  values = {v.name: v for v in (*graph.graph.value_info, *graph.graph.output)}
  if name not in {v.name for v in graph.graph.output}:
    graph.graph.output.append(values[name])
  options = ort.SessionOptions()
  options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
  ref = ort.InferenceSession(graph.SerializeToString(), options, providers=['CPUExecutionProvider'])
  runtimes = {}
  for label, path in (('original_conversion', source), ('lookup_conversion', modified)):
    runner = Interpreter(model_path=str(path), num_threads=2, experimental_preserve_all_tensors=True,
                         experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    runner.allocate_tensors()
    runtimes[label] = runner
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'provider': 'host ONNX CPU and LiteRT CPU; not GPU evidence',
            'original_sha256': expected_hash, 'prefix_sha256': digest(prefix),
            'source_sha256': digest(source), 'modified_sha256': digest(modified), 'seeds': {}}
  for seed in (42, 7, 123):
    inputs = initial_inputs(refs, 'driving_supercombo', seed)
    for key in ('new_img', 'state_img_q'):
      if inputs[key].dtype != np.uint8:
        raise ValueError('Byte-domain input contract changed: ' + key)
    if 'rewritten_source_sha256' in manifest:
      inputs['new_img'] = inputs['new_img'].astype(np.float32)
      inputs['state_img_q'] = inputs['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
    expected = run_onnx(ref, inputs)[name]
    report['seeds'][str(seed)] = {}
    for label, runner in runtimes.items():
      run_converted(runner, {d['name']: inputs[d['name']] for d in runner.get_input_details()})
      details = [t for t in runner.get_tensor_details() if t['name'] == name]
      if len(details) != 1:
        raise ValueError('Normalization output is ambiguous')
      actual, layout = align(runner.get_tensor(details[0]['index']), expected, False)
      report['seeds'][str(seed)][label] = {'element_count': int(actual.size),
        'unequal_elements': int(np.count_nonzero(actual != expected)), 'layout': layout}
  output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for arg in ('sections', 'source', 'modified', 'output'):
    parser.add_argument('--' + arg, type=Path, required=True)
  args = parser.parse_args()
  validate(args.sections, args.source, args.modified, args.output)
