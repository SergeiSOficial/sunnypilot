"""Diagnose original FP16 layer normalization on captured GPU inputs."""
import argparse
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

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate', type=Path, required=True)
p.add_argument('--trace', type=Path, required=True)
p.add_argument('--tensor', default='model/p_layer_norm')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
manifest = json.loads((a.trace / 'manifest.json').read_text())
records = {v['tensor']: v for v in manifest['tensors'] if v['tensor']}
path = ROOT / 'android/app/build/model-reference/host-fusion' / (manifest['model'] + '.onnx')
m = onnx.shape_inference.infer_shapes(onnx.load(path))
n = next(n for n in m.graph.node if a.tensor in n.output)
assert n.op_type == 'LayerNormalization'
info = {v.name: v for v in (*m.graph.input, *m.graph.value_info, *m.graph.output)}
constants = {v.name: v for v in m.graph.initializer}
feeds = {}
for key in n.input:
  if key in constants:
    continue
  rec = records[key]
  feeds[key] = np.fromfile(a.trace / rec['file'], dtype=rec['dtype']).reshape(rec['shape']).astype(
      onnx.helper.tensor_dtype_to_np_dtype(info[key].type.tensor_type.elem_type))
g = onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(n)], 'actual_input_layernorm',
    [info[k] for k in feeds], [info[k] for k in n.output], [constants[k] for k in n.input if k in constants]),
    opset_imports=list(m.opset_import))
g.ir_version = m.ir_version
options = ort.SessionOptions()
options.intra_op_num_threads = 2
options.inter_op_num_threads = 1
expected = ort.InferenceSession(g.SerializeToString(), options, providers=['CPUExecutionProvider']).run(None, feeds)[0]
rec = records[a.tensor]
actual = np.fromfile(a.trace / rec['file'], dtype=rec['dtype']).reshape(rec['shape']).astype(expected.dtype)
attrs = {v.name: onnx.helper.get_attribute_value(v) for v in n.attribute}
assert attrs.get('axis', -1) == -1 and attrs.get('stash_type', 1) == 1
assert expected.dtype == np.float16
x = feeds[n.input[0]].astype(np.float32)
gamma, beta = (onnx.numpy_helper.to_array(constants[k]).astype(np.float32) for k in n.input[1:3])
fmaf = ctypes.CDLL(None).fmaf
fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
fmaf.restype = ctypes.c_float
sqrtf = ctypes.CDLL(None).sqrtf
sqrtf.argtypes = (ctypes.c_float,)
sqrtf.restype = ctypes.c_float
fp = np.float32
result = np.empty_like(x)
for row, dest in zip(x.reshape(-1, x.shape[-1]), result.reshape(-1, x.shape[-1]), strict=True):
  mean, m2 = fp(0), fp(0)
  for i, value in enumerate(row):
    delta = fp(value - mean)
    mean = fp(mean + fp(delta / fp(i + 1)))
    m2 = fp(fmaf(delta, fp(value - mean), m2))
  std = fp(sqrtf(fp(fp(m2 / fp(len(row))) + fp(attrs.get('epsilon', 1e-5)))))
  for i, value in enumerate(row):
    dest[i] = fmaf(fp(fp(value - mean) / std), gamma[i], beta[i])
result = result.astype(np.float16)
artifact = json.loads((a.candidate / 'artifacts.json').read_text())
output = {'candidate': a.candidate.name, 'apk_sha256': artifact['files']['app-debug.apk']['sha256'],
    'diagnostic_only': True, 'node': n.name, 'tensor': a.tensor, 'seed': manifest['seed'],
    'captured_gpu': {'parity': compare(actual, expected), 'unequal_elements': int(np.count_nonzero(actual != expected))},
    'ordered_welford': {'parity': compare(result, expected), 'unequal_elements': int(np.count_nonzero(result != expected))},
    'input_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in feeds.items()},
    'original_host_graph_sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
    'diagnostic_graph_sha256': hashlib.sha256(g.SerializeToString()).hexdigest()}
a.output.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps(output, indent=2))
