#!/usr/bin/env python3
"""Replace one internal suffix value for a host-only causal diagnostic."""
import argparse
import copy
import json
from pathlib import Path

import flatbuffers
import numpy as np
import onnx
import onnxruntime as ort
from ai_edge_litert import schema_py_generated as schema
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from onnx import numpy_helper

from decompose_road import ROOT, digest
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from prepare_snpe_dense_probe import ordered_half_gemv
from trace_litert_prefix import tap_impact
from validate_litert_dense_sections_host import run_converted, run_onnx, session

OVERRIDE = 'fold3_diagnostic_override'


def intervene(source, name, override_name=OVERRIDE):
  model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(source, 0))
  if len(model.subgraphs) != 1:
    raise ValueError('Expected a single subgraph')
  graph = model.subgraphs[0]
  names = [t.name.decode() if isinstance(t.name, bytes) else t.name for t in graph.tensors]
  if not override_name or names.count(name) != 1 or override_name in names:
    raise ValueError('Target missing or ambiguous, or already patched')
  index = names.index(name)
  target = graph.tensors[index]
  if (index in graph.inputs or index in graph.outputs or target.type != schema.TensorType.FLOAT32
      or target.buffer != 0 or target.isVariable or target.shape is None
      or any(int(d) <= 0 for d in target.shape)
      or (target.shapeSignature is not None and any(int(d) <= 0 for d in target.shapeSignature))):
    raise ValueError('Target must be a static internal float32 tensor')
  producers = [op for op in graph.operators if index in op.outputs]
  consumers = [op for op in graph.operators if index in op.inputs]
  if len(producers) != 1 or not consumers:
    raise ValueError('Target must have one producer and at least one consumer')
  override = copy.deepcopy(target)
  override.name = override_name
  new_index = len(graph.tensors)
  graph.tensors.append(override)
  graph.inputs = np.append(graph.inputs, np.int32(new_index)).astype(np.int32)
  for op in consumers:
    op.inputs = np.array([new_index if i == index else i for i in op.inputs], dtype=np.int32)
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
  return bytes(builder.Output())


def run(sections, conversion, model, derived_sha256, output, onnx_names, litert_names):
  if output.exists():
    raise ValueError('Use a new output directory')
  if (not onnx_names or len(onnx_names) != len(litert_names)
      or len(set(onnx_names)) != len(onnx_names) or len(set(litert_names)) != len(litert_names)):
    raise ValueError('Supply matching, unique tensor names')
  manifest = json.loads((sections / 'manifest.json').read_text())
  receipt = json.loads((conversion / 'receipt.json').read_text())
  refs = ROOT / 'android/app/build/model-reference'
  expected_hash = json.loads((refs / 'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if digest(ROOT / 'openpilot/selfdrive/modeld/models/driving_supercombo.onnx') != expected_hash or manifest['original_sha256'] != expected_hash:
    raise ValueError('Original model provenance mismatch')
  for name in ('prefix', 'dense', 'suffix'):
    if digest(sections / (name + '.onnx')) != manifest['sections'][name]['sha256']:
      raise ValueError('Section provenance mismatch')
  if digest(sections / 'suffix.onnx') != receipt['source_sha256'] or digest(model) != derived_sha256:
    raise ValueError('Conversion provenance mismatch')
  graph = onnx.shape_inference.infer_shapes(onnx.load(sections / 'suffix.onnx'))
  infos = {v.name: v for v in graph.graph.value_info}
  if not set(onnx_names) <= infos.keys():
    raise ValueError('ONNX target missing or already public')
  for name in onnx_names:
    graph.graph.output.append(copy.deepcopy(infos[name]))
  options = ort.SessionOptions()
  options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
  tapped = ort.InferenceSession(graph.SerializeToString(), options, providers=['CPUExecutionProvider'])
  source = model.read_bytes()
  changed = source
  overrides = [OVERRIDE if len(litert_names) == 1 else f'{OVERRIDE}_{i}' for i in range(len(litert_names))]
  for name, override_name in zip(litert_names, overrides, strict=True):
    changed = intervene(changed, name, override_name)
  kwargs = {'num_threads': 2, 'experimental_op_resolver_type': OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES}
  original, modified = Interpreter(model_content=source, **kwargs), Interpreter(model_content=changed, **kwargs)
  original.allocate_tensors()
  modified.allocate_tensors()
  details = {d['name']: d for d in modified.get_input_details()}
  dense = onnx.load(sections / 'dense.onnx').graph
  weights = {v.name: numpy_helper.to_array(v) for v in dense.initializer}
  prefix_session, suffix_session = session(sections / 'prefix.onnx'), session(sections / 'suffix.onnx')
  records = {}
  for seed in (42, 7, 123):
    inputs = initial_inputs(refs, 'driving_supercombo', seed)
    if 'rewritten_source_sha256' in manifest:
      inputs['new_img'] = inputs['new_img'].astype(np.float32)
      inputs['state_img_q'] = inputs['state_img_q'].astype(np.float32).reshape(2, 30, 128, 256)
    prefix = run_onnx(prefix_session, inputs)
    inputs.update(prefix)
    inputs['model/linear'] = ordered_half_gemv(prefix['model/view'], *(weights[n] for n in dense.node[0].input[1:]))
    reference = run_onnx(suffix_session, inputs)
    intermediate = run_onnx(tapped, inputs)
    impact = tap_impact(reference, intermediate)
    if not impact['tolerance_preserved']:
      raise ValueError(f'Tapping changed public output beyond tolerance for seed {seed}')
    injected = {}
    for name, override_name in zip(onnx_names, overrides, strict=True):
      value, detail = intermediate[name], details[override_name]
      converted = value.astype(detail['dtype'])
      if (tuple(converted.shape) != tuple(detail['shape']) or not np.isfinite(converted).all()
          or not np.array_equal(converted.astype(value.dtype), value)):
        raise ValueError('Intervention changes tensor values or shape')
      injected[override_name] = converted
    baseline = run_converted(original, {d['name']: inputs[d['name']] for d in original.get_input_details()})
    altered = run_converted(modified, {**{d['name']: inputs[d['name']] for d in original.get_input_details()},
                                       **injected})
    if set(baseline) != set(reference) or set(altered) != set(reference):
      raise ValueError('Output contracts differ')
    records[str(seed)] = {'tap_impact': impact,
                          'baseline': {name: compare(baseline[name], v) for name, v in reference.items()},
                          'intervention': {name: compare(altered[name], v) for name, v in reference.items()}}
  output.mkdir(parents=True)
  (output / 'intervention.tflite').write_bytes(changed)
  report = {'diagnostic_only': True, 'qualified': False, 'provider': 'host LiteRT CPU; not GPU evidence',
            'warning': 'Intervention and ONNX tapping may change graph execution or fusion',
            'tensor_pairs': [{'onnx': a, 'litert': b} for a, b in zip(onnx_names, litert_names, strict=True)],
            'original_sha256': expected_hash,
            'source_sha256': digest(sections / 'suffix.onnx'), 'derived_sha256': derived_sha256,
            'intervention_sha256': digest(output / 'intervention.tflite'), 'seeds': records}
  (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps({s: {k: {n: c['mismatched_elements'] for n, c in r[k].items()}
                        for k in ('baseline', 'intervention')} for s, r in records.items()}, indent=2))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('sections', 'conversion', 'model', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--derived-sha256', required=True)
  parser.add_argument('--onnx-tensor', action='append', required=True)
  parser.add_argument('--litert-tensor', action='append', required=True)
  args = parser.parse_args()
  run(args.sections, args.conversion, args.model, args.derived_sha256, args.output, args.onnx_tensor, args.litert_tensor)
