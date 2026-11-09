"""Execute the shader arithmetic as strict host C++; GPU execution remains separate."""
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'android/app/src/main/python'))
from fold3_model_trace import compare
shader = ROOT / 'android/build/native/accelerators/ncnn-patched/ncnn-7e36d25cb153092ba87a60530455c7376fedf452/src/layer/vulkan/shader/innerproduct.comp'
body = shader.read_text().split('float fold3_half_rte', 1)[1].split('void main()', 1)[0]
body = 'float fold3_half_rte' + body
body = re.sub(r'\bprecise\s+', '', body)
body = re.sub(r'\buint\b', 'uint32_t', body)
body = body.replace('unpackHalf2x16(sign | result).x', 'half_float(sign | result)')
cpp = '''#include <algorithm>
#include <cstdint>
#include <cstring>
using std::min; using std::max; using std::clamp;
static const float *bottom_blob_data, *weight_data, *bias_data;
static int w, outw, bias_term=1;
#define psc(x) (x)
#define buffer_ld1(buf,i) ((buf)[i])
static uint32_t floatBitsToUint(float v) {uint32_t u; memcpy(&u,&v,4);return u;}
static float uintBitsToFloat(uint32_t u) {float v;memcpy(&v,&u,4);return v;}
static float half_float(uint32_t u) {uint16_t h=u;_Float16 v;memcpy(&v,&h,2);return float(v);}
''' + body + '''
extern "C" void run(const float* x, const float* weights, const float* bias, float* output, int k, int n) {
 bottom_blob_data=x; weight_data=weights; bias_data=bias; w=k;outw=n;
 for(int row=0;row<n;row++) output[row]=fold3_gemv(row);
}
'''
source = HERE / 'linear-shader-host.cpp'
source.write_text(cpp)
library = HERE / 'linear-shader-host.dylib'
subprocess.run(['/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++',
 '-std=c++17', '-O2', '-ffp-contract=off', '-dynamiclib', str(source), '-o', str(library)], check=True)
graph_path = ROOT / 'android/app/build/model-reference/accelerators/dmonitoring_model.onnx'
graph = onnx.shape_inference.infer_shapes(onnx.load(graph_path))
info = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
constants = {v.name: v for v in graph.graph.initializer}
run = ctypes.CDLL(str(library)).run
pointer = ctypes.POINTER(ctypes.c_float)
run.argtypes = [pointer,pointer,pointer,pointer,ctypes.c_int,ctypes.c_int]
options = ort.SessionOptions()
options.intra_op_num_threads = 2
checks = []
for node in graph.graph.node:
  if node.op_type != 'Gemm' or constants[node.input[1]].data_type != onnx.TensorProto.FLOAT16:
    continue
  weights = numpy_helper.to_array(constants[node.input[1]])
  bias = numpy_helper.to_array(constants[node.input[2]])
  n, k = weights.shape
  section = helper.make_model(helper.make_graph([copy.deepcopy(node)], 'original-half-gemm',
    [copy.deepcopy(info[node.input[0]])], [copy.deepcopy(info[node.output[0]])],
    [copy.deepcopy(constants[name]) for name in node.input[1:]]),
    opset_imports=list(graph.opset_import), ir_version=graph.ir_version)
  session = ort.InferenceSession(section.SerializeToString(), options, providers=['CPUExecutionProvider'])
  y, b = (np.ascontiguousarray(value, dtype=np.float32) for value in (weights, bias))
  for seed in (42, 7, 123):
    sample = np.random.default_rng(seed).standard_normal((1, k)).astype(np.float16)
    expected = session.run(None, {node.input[0]: sample})[0]
    expanded = sample.astype(np.float32).ravel()
    output = np.empty((1, n), np.float32)
    run(*(a.ctypes.data_as(pointer) for a in (expanded, y, b, output)), k, n)
    checks.append({'node': node.name, 'seed': seed, 'k': k, 'n': n,
      'section_sha256': hashlib.sha256(section.SerializeToString()).hexdigest(), 'parity': compare(output, expected)})
record = {'diagnostic_only': True, 'qualified': False, 'checks': checks,
  'shader_sha256': hashlib.sha256(shader.read_bytes()).hexdigest(),
  'original_graph_sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest(),
  'status': 'passed' if all(c['parity']['passed'] for c in checks) else 'failed'}
(HERE/'half-heads-host-parity.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({'status': record['status'], 'checks': len(checks),
  'failures': [c for c in checks if not c['parity']['passed']]}))
raise SystemExit(record['status'] != 'passed')
