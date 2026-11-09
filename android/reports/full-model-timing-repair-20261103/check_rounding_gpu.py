"""Compare captured-input operations with the unchanged original ONNX sections."""
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'android/app/src/main/python'))
from fold3_model_trace import compare
folder = HERE/'rounding-heads-trace-seed42'
path = ROOT/'android/app/build/model-reference/accelerators/dmonitoring_model.onnx'
graph = onnx.shape_inference.infer_shapes(onnx.load(path))
info = {v.name:v for v in (*graph.graph.input,*graph.graph.value_info,*graph.graph.output)}
options = ort.SessionOptions(); options.intra_op_num_threads=2
checks=[]
# Reuse the already-qualified MNN host-polynomial arithmetic in scalar host form.
cpp='''#include <cmath>
#include <algorithm>
extern "C" void run(const float* input,float* output,int n) {
 for(int i=0;i<n;i++) {
 float x=std::clamp(input[i],-9.f,9.f); float s=x*x;
 float a=-2.76076847742355e-16f;
 a=std::fma(a,s,2.00018790482477e-13f); a=std::fma(a,s,-8.60467152213735e-11f);
 a=std::fma(a,s,5.12229709037114e-08f); a=std::fma(a,s,1.48572235717979e-05f);
 a=std::fma(a,s,6.37261928875436e-04f); a=std::fma(a,s,4.89352455891786e-03f);
 float b=1.19825839466702e-06f; b=std::fma(b,s,1.18534705686654e-04f);
 b=std::fma(b,s,2.26843463243900e-03f); b=std::fma(b,s,4.89352518554385e-03f);
 float product=a*x; output[i]=product/b;
 }
}
'''
source=HERE/'activation-host.cpp';source.write_text(cpp); lib=HERE/'activation-host.dylib'
subprocess.run(['/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++','-std=c++17','-O2','-ffp-contract=off','-dynamiclib',str(source),'-o',str(lib)],check=True)
run=ctypes.CDLL(str(lib)).run; ptr=ctypes.POINTER(ctypes.c_float);run.argtypes=[ptr,ptr,ctypes.c_int]
for operation, before, after in [('ReduceMean',425,426),('Tanh',421,422)]:
 node = (next(n for n in graph.graph.node if n.op_type==operation) if operation=='ReduceMean' else [n for n in graph.graph.node if n.op_type=='Tanh'][-2])
 section=helper.make_model(helper.make_graph([copy.deepcopy(node)],'same-captured-input',[copy.deepcopy(info[node.input[0]])],[copy.deepcopy(info[node.output[0]])],[copy.deepcopy(v) for v in graph.graph.initializer if v.name in node.input]),opset_imports=list(graph.opset_import),ir_version=graph.ir_version)
 session=ort.InferenceSession(section.SerializeToString(),options,providers=['CPUExecutionProvider'])
 shape=tuple(d.dim_value for d in info[node.input[0]].type.tensor_type.shape.dim)
 for seed in ('captured',42,7,123):
  x=(np.fromfile(folder/f'trace-{before}.bin',np.float32).reshape(shape) if seed=='captured' else np.random.default_rng(seed).standard_normal(shape).astype(np.float32))
  expected=session.run(None,{node.input[0]:x})[0]
  if operation=='ReduceMean':
   rows=x.reshape(-1,96); lanes=rows[:,:8].copy()
   for offset in range(8,96,8):lanes=np.add(lanes,rows[:,offset:offset+8],dtype=np.float32)
   packed=np.add(lanes[:,:4],lanes[:,4:],dtype=np.float32)
   summed=np.add(np.add(packed[:,0],packed[:,1]),np.add(packed[:,2],packed[:,3]))
   actual=(summed/np.float32(96)).reshape(expected.shape)
  else:
   actual=np.empty_like(x); run(x.ctypes.data_as(ptr),actual.ctypes.data_as(ptr),x.size)
  r={'operation':operation,'node':node.name,'seed':seed,'section_sha256':hashlib.sha256(section.SerializeToString()).hexdigest(),'parity':compare(actual,expected),'exact_elements':int(np.count_nonzero(actual==expected)),'elements':actual.size}
  if seed=='captured':r['gpu_same_input']=compare(np.fromfile(folder/f'trace-{after}.bin',np.float32).reshape(expected.shape),expected)
  checks.append(r)
record={'status':'passed' if all(c['parity']['passed'] and c.get('gpu_same_input', {'passed':True})['passed'] for c in checks) else 'failed','diagnostic_only':True,'qualified':False,'original_graph_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'checks':checks}
(HERE/'rounding-activation-mean-gpu.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
raise SystemExit(record['status']!='passed')
