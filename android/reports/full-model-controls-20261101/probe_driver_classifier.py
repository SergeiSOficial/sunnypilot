"""Replay the first observed driver half-GEMV violation on its actual GPU input."""
import copy
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
folder = candidate / 'traces288/accel-trace-dmonitoring_model-buffer'
manifest = json.loads((folder / 'manifest.json').read_text())
graph = onnx.shape_inference.infer_shapes(onnx.load(candidate / 'graphs/dmonitoring_model-mnn-io.onnx'))
key = 'fold3_half_gemv8_346_0'
node = next(node for node in graph.graph.node if key in node.output)
records = {record['tensor']: record for record in manifest['tensors']}
def captured(record):
  return np.frombuffer((folder / record['file']).read_bytes(), dtype=record['dtype']).reshape(record['shape'])
values = captured(records['gelu_29']).reshape(1, 1280)
assert np.array_equal(values, values.astype(np.float16).astype(np.float32))
actual = captured(records[key])
info = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
initializers = {value.name: value for value in graph.graph.initializer}
local = onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(node)], 'actual_gpu_input_half_classifier',
    [info[node.input[0]]], [info[key]], [copy.deepcopy(initializers[name]) for name in node.input[1:]]),
    opset_imports=list(graph.opset_import))
local.ir_version = graph.ir_version
options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
expected = ort.InferenceSession(local.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, {node.input[0]: values.astype(np.float16)})[0]
result = {'candidate': 'candidate45', 'apk_sha256': json.loads((candidate / 'artifacts.json').read_text())['files']['app-debug.apk']['sha256'],
    'diagnostic_only': True, 'operator': node.op_type, 'original_node': node.name, 'tensor': key,
    'shape': list(actual.shape), 'semantic_dtype': 'float16', 'physical_gpu_dtype': str(actual.dtype),
    'actual_gpu_input_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
    'diagnostic_graph_sha256': hashlib.sha256(local.SerializeToString()).hexdigest(),
    'parity_on_actual_input': compare(actual, expected),
    'note': 'This local actual-input comparison does not qualify full-model accuracy.'}
output = Path(__file__).with_name('driver45-classifier-actual-input.json')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
