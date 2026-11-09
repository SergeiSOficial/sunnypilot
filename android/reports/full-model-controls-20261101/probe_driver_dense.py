"""Replay an observed driver dense mismatch with the captured GPU input."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'android/app/src/main/python'))
from fold3_model_trace import compare
folder = ROOT/'android/build/candidates/candidate45/traces352/accel-trace-dmonitoring_model-buffer'
manifest = json.loads((folder/'manifest.json').read_text())
graph = onnx.load(ROOT/'android/build/candidates/candidate45/graphs/dmonitoring_model-mnn-io.onnx')
node = next(n for n in graph.graph.node if 'android_precision_335_result' in n.output)
initializers = {v.name: v for v in graph.graph.initializer}
input_record, output_record = manifest['tensors'][:2]
def captured(record):
  return np.frombuffer((folder/record['file']).read_bytes(), dtype=record['dtype']).reshape(record['shape'])
values = captured(input_record).reshape(1, 24)
actual = captured(output_record).reshape(1, 16)
relu = onnx.helper.make_node('Relu', [node.output[0]], ['local_relu'])
local = onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(node), relu], 'actual_gpu_input_gemm_relu',
    [onnx.helper.make_tensor_value_info(node.input[0], onnx.TensorProto.FLOAT, [1, 24])],
    [onnx.helper.make_tensor_value_info('local_relu', onnx.TensorProto.FLOAT, [1, 16])],
    [copy.deepcopy(initializers[k]) for k in node.input[1:]]), opset_imports=list(graph.opset_import))
local.ir_version = graph.ir_version
options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
expected = ort.InferenceSession(local.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, {node.input[0]: values})[0]
weights, bias = [onnx.numpy_helper.to_array(initializers[k]) for k in node.input[1:]]
float64 = np.maximum(0, (values.astype(np.float64) @ weights.astype(np.float64).T + bias.astype(np.float64)).astype(np.float32))
result = {'candidate': 'candidate45', 'apk_sha256': '9868e5b9ca7d24feaefd9acfbcdabaf4f5226e85063e03fffec4400aecc1a859',
    'diagnostic_only': True, 'original_node': node.name, 'operator': 'Gemm+Relu (MNN fusion)', 'graph_tensor_before_fused_relu': node.output[0], 'captured_tensor_after_fused_relu': output_record['tensor'],
    'actual_gpu_input_sha256': hashlib.sha256(values.tobytes()).hexdigest(),
    'diagnostic_graph_sha256': hashlib.sha256(local.SerializeToString()).hexdigest(),
    'parity_on_actual_input': compare(actual, expected),
    'gpu_vs_float64_diagnostic': compare(actual, float64),
    'host_vs_float64_diagnostic': compare(expected, float64),
    'note': 'Actual-input local agreement cannot qualify full-model output or locate an earlier upstream error.'}
output = Path(__file__).with_name('driver45-dense335-actual-input.json')
output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
