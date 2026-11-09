"""Test trace consistency with the pinned optimized graph and fusion disabled."""
import copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'android/app/src/main/python'))
from fold3_sequences import initial_inputs
from fold3_model_trace import compare
base=ROOT/'android/app/build/model-reference'
path=base/'host-fusion/dmonitoring_model.onnx'
m=onnx.load(path)
o=ort.SessionOptions();o.intra_op_num_threads=2;o.graph_optimization_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL
inputs=initial_inputs(base,'dmonitoring_model',42)
report={'diagnostic_only':True,'qualified':False,'graph_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
try:
 expected=np.load(base/'dmonitoring_model-outputs.npz')['outputs']
 actual=ort.InferenceSession(m.SerializeToString(),o,providers=['CPUExecutionProvider']).run(None,inputs)[0]
 report['untapped_parity']=compare(actual,expected)
 typed=onnx.shape_inference.infer_shapes(onnx.load(base/'accelerators/dmonitoring_model.onnx'))
 info={v.name:v for v in (*typed.graph.input,*typed.graph.value_info,*typed.graph.output)}
 report['taps']={}
 for name in ['InsertedPrecisionFreeCast_relu_26','relu_26','linear_36']:
  tapped=copy.deepcopy(m)
  tapped.graph.output.append(info[name])
  values=ort.InferenceSession(tapped.SerializeToString(),o,providers=['CPUExecutionProvider']).run(None,inputs)
  report['taps'][name]={'final_output_unchanged':bool(np.array_equal(values[0],actual)),'parity':compare(values[0],expected),'sha256':hashlib.sha256(values[-1].tobytes()).hexdigest()}
  np.save(Path(__file__).parent/(name+'.npy'),values[-1])
except Exception as e:report['error']=str(e)
(Path(__file__).parent/'frozen-reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
