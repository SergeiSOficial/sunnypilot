import sys,json,copy
from pathlib import Path
import numpy as np
import onnx,onnxruntime as ort
root=Path.cwd();sys.path.insert(0,str(root/'android/scripts'))
from ncnn_replay_inputs import replay_case
base=root/'android/app/build/model-reference';folder=root/'android/reports/final-head-20261104/replay21-projection'
r=json.loads(folder.with_name(folder.name+'-comparison.json').read_text()); target=next(c['original_tensor'] for c in r['checks'] if c['operator']=='pow_224')
m=onnx.load(base/'host-fusion/dmonitoring_model.onnx');n=next(n for n in m.graph.node if target in n.output)
typed=onnx.shape_inference.infer_shapes(onnx.load(base/'accelerators/dmonitoring_model.onnx'))
info={v.name:v for v in (*typed.graph.input,*typed.graph.value_info,*typed.graph.output)}
for name in [n.input[0],target]:m.graph.output.append(copy.deepcopy(info[name]))
o=ort.SessionOptions();o.intra_op_num_threads=2;o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
inputs,golden=replay_case(base,21);actual,x,y=ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider']).run(None,inputs)
assert np.array_equal(actual,golden)
gpu=np.fromfile(folder/'trace-431.bin',np.float32).reshape(y.shape)
candidates={'gpu':gpu,'multiply':(x*x)*x,'double':(x.astype(np.float64)**3).astype(np.float32),'numpy':np.power(x,np.float32(3))}
result={'node':n.name,'shape':list(x.shape),'matches':{k:int(np.count_nonzero(v==y)) for k,v in candidates.items()},'elements':x.size}
ids=np.flatnonzero(gpu!=y);result['examples']=[{'index':int(i),'x':float(x.flat[i]),'expected':float(y.flat[i]),**{k:float(v.flat[i]) for k,v in candidates.items()}} for i in ids[:8]]
np.savez(folder.parent/'cube-captured.npz',x=x,y=y,gpu=gpu)
print(json.dumps(result,indent=2))
