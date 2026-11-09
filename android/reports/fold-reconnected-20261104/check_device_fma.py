"""Compare captured Vulkan projection arithmetic with fused/separate CPU probes."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import onnx
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = HERE / 'projection-arithmetic.cpp'
source.write_text('''#include <cmath>
#include <cstring>
static float high(float x){unsigned u;std::memcpy(&u,&x,4);u&=0xfffff000u;std::memcpy(&x,&u,4);return x;}
static float compensated(float a,float b,float c){float ah=high(a),bh=high(b),al=a-ah,bl=b-bh;float p=a*b;float e=((ah*bh-p)+ah*bl+al*bh)+al*bl;float s=p+c;float v=s-p;float se=(p-(s-v))+(c-v);return s+(se+e);}
extern "C" void run(const float*x,const float*w,const float*b,float*y,int k,int n,int s,int mode){for(int o=0;o<n;o++)for(int p=0;p<s;p++){float sum=0;for(int i=0;i<k;i++){if(mode==0)sum=std::fma(x[i*s+p],w[o*k+i],sum);else if(mode==2)sum=compensated(x[i*s+p],w[o*k+i],sum);else {volatile float product=x[i*s+p]*w[o*k+i];sum=sum+product;}}y[o*s+p]=b?sum+b[o]:sum;}}
''')
lib = HERE / 'projection-arithmetic.dylib'
subprocess.run(['/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++', '-O2', '-ffp-contract=off', '-dynamiclib', str(source), '-o', str(lib)], env=dict(os.environ, SDKROOT='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk'), check=True)
run = ctypes.CDLL(str(lib)).run
ptr = ctypes.POINTER(ctypes.c_float)
run.argtypes = [ptr, ptr, ptr, ptr, *([ctypes.c_int] * 4)]
graph = onnx.shape_inference.infer_shapes(onnx.load(ROOT / 'android/app/build/model-reference/accelerators/dmonitoring_model.onnx'))
info = {v.name:v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
const = {v.name:onnx.numpy_helper.to_array(v) for v in graph.graph.initializer}
folder = HERE / 'physical-prefix-trace'
trace = json.loads(folder.with_suffix('.json').read_text())
tensors = {t['blob']:t for t in trace['tensors']}
checks = []
for operator, node_name in [('conv_97','node_Conv_891'), ('conv_98','node_Conv_893'), ('conv_99','node_Conv_895')]:
    node = next(n for n in graph.graph.node if n.name == node_name)
    words = next(l.split() for l in (folder/'model.param').read_text().splitlines() if operator in l.split())
    shape = tuple(d.dim_value for d in info[node.input[0]].type.tensor_type.shape.dim)
    x = np.fromfile(folder/tensors[words[4]]['file'], np.float32).reshape(shape)
    w = np.ascontiguousarray(const[node.input[1]], dtype=np.float32)
    b = np.ascontiguousarray(const[node.input[2]], dtype=np.float32) if len(node.input)>2 else None
    observed = np.fromfile(folder/tensors[words[5]]['file'], np.float32)
    check = {'operator':operator, 'elements':int(observed.size)}
    for mode, name in enumerate(('fused', 'separate', 'compensated')):
        actual = np.empty_like(observed)
        run(x.ctypes.data_as(ptr), w.ctypes.data_as(ptr), b.ctypes.data_as(ptr) if b is not None else None, actual.ctypes.data_as(ptr), shape[1], w.shape[0], int(x.size/shape[1]), mode)
        if mode == 0:
            fused = actual.copy()
        if mode == 2:
            check['compensated_vs_fused'] = {'exact_elements':int(np.count_nonzero(actual==fused)), 'max_abs_error':float(np.max(np.abs(actual-fused)))}
        check[name] = {'exact_elements':int(np.count_nonzero(actual==observed)), 'max_abs_error':float(np.max(np.abs(actual-observed)))}
    checks.append(check)
(HERE/'projection-arithmetic.json').write_text(json.dumps(checks,indent=2)+'\n')
print(checks)
