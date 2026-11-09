#!/usr/bin/env python3
"""Export independent SNPE primitives with original ONNX reference outputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
import onnxruntime as ort

from prepare_htp_probes import graph
from decompose_road import digest


def prepare(destination):
  destination.mkdir(parents=True, exist_ok=True)
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'rtol': 1e-3, 'atol': 1e-4, 'validation_seed': 2029, 'probes': {}}
  for half in (False, True):
    for kind in ('conv', 'gemm', 'activation', 'layout', 'cast'):
      name = ('fp16-' if half else 'fp32-') + kind
      if kind == 'cast':
        if half: continue
        shape = [1, 8]
        model = h.make_model(h.make_graph([
          h.make_node('Cast', ['x'], ['rounded'], to=T.FLOAT16),
          h.make_node('Cast', ['rounded'], ['y'], to=T.FLOAT)], 'explicit_rounding',
          [h.make_tensor_value_info('x', T.FLOAT, shape)], [h.make_tensor_value_info('y', T.FLOAT, shape)]),
          opset_imports=[h.make_opsetid('', 21)], ir_version=10)
        x = np.array([[1.0003, -.333333, .123456, -2.0007, 3.14159, .00123456, 9.003, -.987654]], np.float32)
      else:
        model, shape = graph(kind, half)
        x = np.random.default_rng(2029).uniform(-.2, .2, shape).astype(np.float16 if half else np.float32)
      path = destination/(name+'.onnx'); onnx.save(model, path)
      options = ort.SessionOptions(); options.intra_op_num_threads = 2
      expected = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider']).run(None, {'x': x})[0]
      np.savez(destination/(name+'.npz'), x=x, expected=expected)
      # SNPE's default ITensor interface uses floats even for an FP16 graph.
      # Keep both explicitly identified layouts; select using actual DLC contracts.
      raw = destination/(name+'-nchw.raw'); x.astype(np.float32).tofile(raw)
      entry = {'onnx_sha256': digest(path), 'reference_sha256': digest(destination/(name+'.npz')),
               'input_dtype': str(x.dtype), 'input_shape': list(x.shape),
               'output_dtype': str(expected.dtype), 'output_shape': list(expected.shape),
               'raw_nchw_sha256': digest(raw), 'operator': kind}
      if x.ndim == 4:
        raw = destination/(name+'-nhwc.raw'); x.transpose(0, 2, 3, 1).astype(np.float32).tofile(raw)
        entry['raw_nhwc_sha256'] = digest(raw)
      report['probes'][name] = entry
  (destination/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
  return report


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  print(json.dumps({'probes': len(prepare(args.output)['probes'])}))
