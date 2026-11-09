import copy,ctypes,json,subprocess,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).parent
sys.path.insert(0,str(ROOT/'android/app/src/main/python'))
from fold3_model_trace import compare
p=HERE/'conv99.cpp';p.write_text('''#include <cmath>
extern "C" void run(const float*x,const float*w,const float*b,float*y,int k,int n){for(int o=0;o<n;o++){float sum=0;for(int i=0;i<k;i++)sum=std::fma(x[i],w[o*k+i],sum);y[o]=sum+b[o];}}
''')
lib=HERE/'conv99.dylib';subprocess.run(['/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++','-O2','-ffp-contract=off','-dynamiclib',str(p),'-o',str(lib)],check=True)
run=ctypes.CDLL(str(lib)).run;ptr=ctypes.POINTER(ctypes.c_float);run.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int]
m=onnx.shape_inference.infer_shapes(onnx.load(ROOT/'android/app/build/model-reference/accelerators/dmonitoring_model.onnx'));node=next(n for n in m.graph.node if n.name=='node_Conv_895');info={v.name:v for v in (*m.graph.input,*m.graph.value_info,*m.graph.output)};const={v.name:v for v in m.graph.initializer};w,b=[np.ascontiguousarray(onnx.numpy_helper.to_array(const[n])) for n in node.input[1:]]
section=onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(node)],'same-input',[info[node.input[0]]],[info[node.output[0]]],[const[n] for n in node.input[1:]]),opset_imports=list(m.opset_import),ir_version=m.ir_version);opts=ort.SessionOptions();opts.intra_op_num_threads=2;session=ort.InferenceSession(section.SerializeToString(),opts,providers=['CPUExecutionProvider']);checks=[]
for seed in ['captured',42,7,123]:
 x=np.fromfile(ROOT/'android/reports/full-model-timing-repair-20261103/rounding-heads-trace-seed42/trace-426.bin',np.float32) if seed=='captured' else np.random.default_rng(seed).standard_normal(960).astype(np.float32)
 y=np.empty(1280,np.float32);run(*(a.ctypes.data_as(ptr) for a in [x,w,b,y]),960,1280);expected=session.run(None,{node.input[0]:x.reshape(1,960,1,1)})[0].ravel();checks.append({'seed':seed,'parity':compare(y,expected),'exact_elements':int(np.count_nonzero(y==expected))})
(HERE/'conv99-host.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
