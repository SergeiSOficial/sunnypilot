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
graph_path = ROOT / 'android/app/build/model-reference/accelerators/dmonitoring_model-mnn-io.onnx'
graph = onnx.shape_inference.infer_shapes(onnx.load(graph_path))
node = next(n for n in graph.graph.node if n.name == 'node_linear')
info = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
constants = [copy.deepcopy(v) for v in graph.graph.initializer if v.name in node.input]
weights = {v.name: numpy_helper.to_array(v) for v in constants}
assert {a.name: helper.get_attribute_value(a) for a in node.attribute}.get('transB', 0) == 1
section = helper.make_model(helper.make_graph([copy.deepcopy(node)], 'shader-same-captured-input',
 [copy.deepcopy(info[node.input[0]])], [copy.deepcopy(info[node.output[0]])], constants),
 opset_imports=list(graph.opset_import), ir_version=graph.ir_version)
options = ort.SessionOptions()
options.intra_op_num_threads = 2
capture = ROOT / 'android/reports/gpu-runtime-recovery-20261103/ncnn-driver-seed42-reduction/trace-440.bin'
xhalf = np.fromfile(capture, np.float32).astype(np.float16).reshape(1,1280)
session = ort.InferenceSession(section.SerializeToString(),options,providers=['CPUExecutionProvider'])
expected = session.run(None,{node.input[0]: xhalf})[0]
x = xhalf.astype(np.float32).ravel()
y = np.ascontiguousarray(weights[node.input[1]],dtype=np.float32)
bias = np.ascontiguousarray(weights[node.input[2]],dtype=np.float32)
output = np.empty((1,512),np.float32)
run = ctypes.CDLL(str(library)).run
pointer = ctypes.POINTER(ctypes.c_float)
run.argtypes = [pointer,pointer,pointer,pointer,ctypes.c_int,ctypes.c_int]
run(*(a.ctypes.data_as(pointer) for a in (x,y,bias,output)),1280,512)
record = {'diagnostic_only':True,'gpu_execution_validated':False,'qualified':False,
 'shader_sha256':hashlib.sha256(shader.read_bytes()).hexdigest(),
 'derived_section_sha256':hashlib.sha256(section.SerializeToString()).hexdigest(),
 'captured_input_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),
 'parity':compare(output,expected)}
record['additional_inputs'] = []
for seed in (42, 1337, 2026):
  sample = np.random.default_rng(seed).standard_normal((1,1280)).astype(np.float16)
  expected_sample = session.run(None, {node.input[0]: sample})[0]
  expanded = sample.astype(np.float32).ravel()
  run(*(a.ctypes.data_as(pointer) for a in (expanded,y,bias,output)),1280,512)
  record['additional_inputs'].append({'seed':seed,'parity':compare(output,expected_sample)})
(HERE/'linear-shader-host-parity.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
if not record['parity']['passed'] or any(not item['parity']['passed'] for item in record['additional_inputs']):
 raise SystemExit(1)
