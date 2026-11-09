#!/usr/bin/env python3
"""Repair demonstrated SNPE rank-five Slice and constant GatherND failures."""
import argparse
import json
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
from prepare_accel_models import pack_road_image_history, lower_contiguous_gather_nd, check_original, float_image_transport
from decompose_road import ROOT, digest, split
from reference_precision import portable


def prepare(output, float_transport=False):
  output.mkdir(parents=True, exist_ok=True)
  source = ROOT/'openpilot/selfdrive/modeld/models/driving_supercombo.onnx'
  references = ROOT/'android/app/build/model-reference'
  expected = json.loads((references/'manifest.json').read_text())['models']['driving_supercombo']['sha256']
  if digest(source) != expected: raise ValueError('Original road hash mismatch')
  fusion = references/'host-fusion/driving_supercombo.onnx'
  model = pack_road_image_history(lower_contiguous_gather_nd(portable(onnx.load(fusion), promote_half_gemm=False)))
  # SNPE's Expand lowering creates mixed FP16/FP32 operands. Tile performs
  # the same broadcast by copying bytes, with no added arithmetic boundary.
  model = onnx.shape_inference.infer_shapes(model)
  shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  for node in model.graph.node:
    if node.op_type != 'Expand' or node.name not in ('model/node_expand', 'model/node_expand_1'): continue
    before, after = shapes[node.input[0]], shapes[node.output[0]]
    if len(before) != len(after) or any(a <= 0 or b <= 0 or (a != b and a != 1) for a,b in zip(before,after)):
      raise ValueError('Unproven expansion geometry')
    name = node.name+'/tile_repeats'
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array([b//a for a,b in zip(before,after)], np.int64), name))
    node.CopyFrom(onnx.helper.make_node('Tile', [node.input[0], name], node.output, name=node.name))
  if float_transport:
    model = float_image_transport(model)
  path = output/'road-packed.onnx'; onnx.save(model, path)
  options = ort.SessionOptions(); options.intra_op_num_threads = 2
  session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
  result = check_original(session, references, 'driving_supercombo', packed_history=True,
                          external_outputs={'next_state_img_q': {'type': 'tensor(uint8)'}} if float_transport else None)
  report = {'original_sha256': expected, 'host_fusion_sha256': digest(fusion), 'derived_sha256': digest(path), 'host_reference': result,
            'qualified': False, 'vehicle_tx_enabled': False,
            'rewrites': ['byte-order-preserving history packing', 'contiguous constant GatherND to Slice', 'proven unit-axis Expand to byte-copy Tile']}
  report['float_image_transport'] = float_transport
  (output/'repair.json').write_text(json.dumps(report, indent=2)+'\n')
  if not result['passed'] or result['comparisons'] != 636: raise ValueError('Original reference failed')
  sections, manifest = split(model)
  for name, graph in sections: onnx.save(graph, output/(name+'.onnx'))
  (output/'sections.json').write_text(json.dumps(manifest, indent=2)+'\n')
  print('PASS', result['comparisons'], flush=True)


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
  p.add_argument('--float-image-transport', action='store_true')
  args = p.parse_args(); prepare(args.output, args.float_image_transport)
