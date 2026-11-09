"""Compare actual captured Tanh inputs against pinned host arithmetic variants."""
import ctypes, hashlib, json, os, subprocess
from pathlib import Path
import numpy as np
import onnx, onnxruntime as ort
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
s=HERE/'tanh-arithmetic.cpp'
s.write_text(r"""#include <cmath>
#include <cstring>
static float high(float x){unsigned u;std::memcpy(&u,&x,4);u&=0xfffff000u;std::memcpy(&x,&u,4);return x;}
static float mad(float a,float b,float c,int m){if(m==0)return std::fma(a,b,c); if(m==1){volatile float p=a*b;return p+c;}float ah=high(a),bh=high(b),al=a-ah,bl=b-bh,p=a*b,e=((ah*bh-p)+ah*bl+al*bh)+al*bl,s=p+c,v=s-p,se=(p-(s-v))+(c-v);return s+(se+e);}
extern "C" void run(const float*x,float*y,int n,int m){for(int i=0;i<n;i++){float v=std::fmin(9.f,std::fmax(-9.f,x[i])),q=v*v,a=-2.76076847742355e-16f;
a=mad(a,q,2.00018790482477e-13f,m);a=mad(a,q,-8.60467152213735e-11f,m);a=mad(a,q,5.12229709037114e-08f,m);a=mad(a,q,1.48572235717979e-05f,m);a=mad(a,q,6.37261928875436e-04f,m);a=mad(a,q,4.89352455891786e-03f,m);
float b=1.19825839466702e-06f;b=mad(b,q,1.18534705686654e-04f,m);b=mad(b,q,2.26843463243900e-03f,m);b=mad(b,q,4.89352518554385e-03f,m);y[i]=(a*v)/b;}}
""")
lib=HERE/'tanh-arithmetic.dylib'
subprocess.run(['/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++','-O2','-ffp-contract=off','-dynamiclib',str(s),'-o',str(lib)],env=dict(os.environ,SDKROOT='/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk'),check=True)
fn=ctypes.CDLL(str(lib)).run; ptr=ctypes.POINTER(ctypes.c_float); fn.argtypes=[ptr,ptr,ctypes.c_int,ctypes.c_int]
folder=ROOT/'android/reports/fold-reconnected-20261104/physical-compensated-trace'
manifest=json.loads(folder.with_suffix('.json').read_text()); ts={t['blob']:t for t in manifest['tensors']}
checks=[]; opts=ort.SessionOptions(); opts.intra_op_num_threads=2
for l in (folder/'model.param').read_text().splitlines()[2:]:
 w=l.split()
 if w[0]!='TanH' or w[4] not in ts or w[5] not in ts:continue
 def read(name):
  t=ts[name]; raw=(folder/t['file']).read_bytes(); assert hashlib.sha256(raw).hexdigest()==t['sha256']; return np.frombuffer(raw,np.float32)
 x=read(w[4]); observed=read(w[5]); vi=onnx.helper.make_tensor_value_info
 graph=onnx.helper.make_model(onnx.helper.make_graph([onnx.helper.make_node('Tanh',['x'],['y'])],'same-input',[vi('x',1,[len(x)])],[vi('y',1,[len(x)])]),opset_imports=[onnx.helper.make_opsetid('',17)],ir_version=9)
 expected=ort.InferenceSession(graph.SerializeToString(),opts,providers=['CPUExecutionProvider']).run(None,{'x':x})[0]
 row={'operator':w[1],'elements':x.size,'gpu_exact_host':int(np.count_nonzero(observed==expected))}
 for m,name in enumerate(('fused','separate','compensated')):
  y=np.empty_like(x);fn(x.ctypes.data_as(ptr),y.ctypes.data_as(ptr),x.size,m)
  row[name]={'exact_host':int(np.count_nonzero(y==expected)),'exact_gpu':int(np.count_nonzero(y==observed)),'max_host_error':float(np.max(np.abs(y-expected)))}
 checks.append(row)
(HERE/'tanh-arithmetic.json').write_text(json.dumps(checks,indent=2)+'\n');print(checks)
