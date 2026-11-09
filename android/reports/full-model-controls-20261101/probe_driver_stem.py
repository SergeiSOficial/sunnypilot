"""Compare the first captured Conv against an isolated graph and ordered FMAs.

This diagnostic records rounding evidence; sampled FMAs cannot qualify a model.
"""
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'android/app/src/main/python'))
from fold3_model_trace import compare

candidate = ROOT / 'android/build/candidates/candidate45'
folder = candidate.parent / 'candidate45h/traces0/accel-trace-dmonitoring_model-buffer'
manifest = json.loads((folder / 'manifest.json').read_text())
graph = onnx.shape_inference.infer_shapes(onnx.load(candidate / 'graphs/dmonitoring_model-mnn-io.onnx'))
key = 'android_precision_8_result'
node = next(node for node in graph.graph.node if key in node.output)
input_record, output_record = manifest['tensors'][4:6]

def captured(record):
  return np.frombuffer((folder / record['file']).read_bytes(), dtype=record['dtype']).reshape(record['shape'])

values, actual = captured(input_record), captured(output_record)
initializers = {value.name: value for value in graph.graph.initializer}
info = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
local = onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(node)], 'actual_gpu_input_stem',
    [info[node.input[0]]], [info[key]], [copy.deepcopy(initializers[name]) for name in node.input[1:]]),
    opset_imports=list(graph.opset_import))
local.ir_version = graph.ir_version
options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
expected = ort.InferenceSession(local.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, {node.input[0]: values})[0]
weights, bias = [onnx.numpy_helper.to_array(initializers[name]) for name in node.input[1:]]
attributes = {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
assert attributes['group'] == 1 and attributes['dilations'] == [1, 1]
assert values.shape[0] == 1 and weights.shape[1] == values.shape[1]
fma = ctypes.CDLL(None).fmaf
fma.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
fma.restype = ctypes.c_float
positions = np.argpartition(np.abs(actual - expected).reshape(-1), -12)[-12:]
samples = []
for position in positions:
  batch, channel, y, x = np.unravel_index(int(position), actual.shape)
  variants = {}
  for order in ('ky_kx_ci', 'ci_ky_kx'):
    terms = [(ky, kx, ci) for ky in range(weights.shape[2]) for kx in range(weights.shape[3]) for ci in range(weights.shape[1])]
    if order == 'ci_ky_kx':
      terms.sort(key=lambda item: (item[2], item[0], item[1]))
    for bias_first in (False, True):
      accumulator = float(bias[channel]) if bias_first else 0.
      for ky, kx, ci in terms:
        iy = y * attributes['strides'][0] + ky - attributes['pads'][0]
        ix = x * attributes['strides'][1] + kx - attributes['pads'][1]
        value = float(values[batch, ci, iy, ix]) if 0 <= iy < values.shape[2] and 0 <= ix < values.shape[3] else 0.
        accumulator = fma(value, float(weights[channel, ci, ky, kx]), accumulator)
      if not bias_first:
        accumulator = float(np.float32(accumulator) + bias[channel])
      variants[order + ('_bias_first' if bias_first else '_bias_last')] = accumulator
  samples.append({'index': [int(v) for v in (batch, channel, y, x)],
      'actual': float(actual[batch, channel, y, x]), 'host': float(expected[batch, channel, y, x]), 'ordered_fma': variants})
result = {'candidate': 'candidate45', 'apk_sha256': json.loads((candidate / 'artifacts.json').read_text())['files']['app-debug.apk']['sha256'],
    'diagnostic_only': True, 'original_node': node.name, 'operator': node.op_type,
    'actual_gpu_input_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
    'diagnostic_graph_sha256': hashlib.sha256(local.SerializeToString()).hexdigest(),
    'parity_on_actual_input': compare(actual, expected), 'largest_error_samples': samples,
    'note': 'Local Conv parity and sampled accumulation orders do not qualify the complete model.'}
output = Path(__file__).with_name('driver45-stem-actual-input.json')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'parity': result['parity_on_actual_input'], 'samples': len(samples)}))
