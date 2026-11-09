"""Check the proposed ordered 1x1 arithmetic against original same-input sections."""
import copy
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import onnx
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[3]; HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'android/app/src/main/python'))
from fold3_model_trace import compare
source=HERE/'depthwise-prefix.cpp'
source.write_text('#include <cmath>\nextern "C" void run(const float*x,const float*w,const float*b,float*y,int c,int h,int width,int kernel){int pad=kernel/2;for(int o=0;o<c;o++)for(int yy=0;yy<h;yy++)for(int xx=0;xx<width;xx++){float sum=0;for(int ky=0;ky<kernel;ky++)for(int kx=0;kx<kernel;kx++){int iy=yy+ky-pad,ix=xx+kx-pad;float v=iy<0||iy>=h||ix<0||ix>=width?0:x[(o*h+iy)*width+ix];sum=std::fma(v,w[(o*kernel+ky)*kernel+kx],sum);}y[(o*h+yy)*width+xx]=sum+b[o];}}\n')
lib=HERE/'depthwise-prefix.dylib'
env=dict(os.environ,SDKROOT='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk')
subprocess.run(['/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++','-O2','-ffp-contract=off','-dynamiclib',str(source),'-o',str(lib)],env=env,check=True)
run=ctypes.CDLL(str(lib)).run; ptr=ctypes.POINTER(ctypes.c_float); run.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int]
m=onnx.shape_inference.infer_shapes(onnx.load(ROOT/'android/app/build/model-reference/accelerators/dmonitoring_model.onnx'))
info={v.name:v for v in (*m.graph.input,*m.graph.value_info,*m.graph.output)}; const={v.name:v for v in m.graph.initializer}
trace=HERE/'pointwise-device-trace'; manifest=json.loads(trace.with_suffix('.json').read_text()); tensors={t['blob']:t for t in manifest['tensors']}
checks=[]
for operator,node_name in [(c['operator'],c['original_node']) for c in json.loads(trace.with_name(trace.name+'-same-input.json').read_text())['checks'] if c['operator'].startswith('convdw_')]:
 n=next(n for n in m.graph.node if n.name==node_name); line=next(l.split() for l in (trace/'model.param').read_text().splitlines() if operator in l)
 shape=tuple(d.dim_value for d in info[n.input[0]].type.tensor_type.shape.dim)
 x=np.fromfile(trace/tensors[line[4]]['file'],np.float32).reshape(shape)
 w=np.ascontiguousarray(onnx.numpy_helper.to_array(const[n.input[1]])); b=np.ascontiguousarray(onnx.numpy_helper.to_array(const[n.input[2]])) if len(n.input)>2 else None
 section=onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(n)],'same-input',[info[n.input[0]]],[info[n.output[0]]],[const[k] for k in n.input[1:]]),opset_imports=list(m.opset_import),ir_version=m.ir_version)
 opts=ort.SessionOptions();opts.intra_op_num_threads=2; session=ort.InferenceSession(section.SerializeToString(),opts,providers=['CPUExecutionProvider'])
 for seed in ['captured']:
  xx=x if seed=='captured' else np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
  expected=session.run(None,{n.input[0]:xx})[0]; actual=np.empty_like(expected)
  run(xx.ctypes.data_as(ptr),w.ctypes.data_as(ptr),b.ctypes.data_as(ptr) if b is not None else None,actual.ctypes.data_as(ptr),shape[1],shape[2],shape[3],w.shape[2])
  checks.append({'operator':operator,'seed':seed,'parity':compare(actual,expected),'exact_elements':int(np.count_nonzero(actual==expected)),'elements':actual.size})
(HERE/'depthwise-prefix-host.json').write_text(json.dumps(checks,indent=2)+'\n')
print([(c['operator'],c['seed'],c['exact_elements'],c['elements'],c['parity']['passed']) for c in checks])
