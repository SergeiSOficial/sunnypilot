"""Replay mapped FP32 operators on actual captured GPU inputs; diagnosis only."""
import argparse, copy, hashlib, json, sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'android/app/src/main/python'))
from fold3_model_trace import compare
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--candidate',type=Path,required=True)
p.add_argument('--trace',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
m=json.loads((a.trace/'manifest.json').read_text())
g=onnx.shape_inference.infer_shapes(onnx.load(a.candidate/'graphs'/(m['model']+'-mnn-io.onnx')))
records={r['tensor']:r for r in m['tensors'] if r['tensor']}
info={v.name:v for v in (*g.graph.input,*g.graph.value_info,*g.graph.output)}
constants={v.name:v for v in g.graph.initializer}
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
checks=[]
for node in g.graph.node:
 if node.op_type not in {'Add','Mul','Sub','Div','Pow','Tanh','Conv','Gemm','ReduceMean'} or len(node.output)!=1 or node.output[0] not in records:continue
 names=list(dict.fromkeys(key for key in node.input if key and key not in constants))
 if not names or not all(key in records and key in info and info[key].type.tensor_type.elem_type==onnx.TensorProto.FLOAT for key in names):continue
 output=node.output[0]
 if output not in info or info[output].type.tensor_type.elem_type!=onnx.TensorProto.FLOAT:continue
 feeds={key:np.fromfile(a.trace/records[key]['file'],dtype=records[key]['dtype']).reshape(records[key]['shape']) for key in names}
 local=onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(node)],'actual_gpu_inputs',[copy.deepcopy(info[key]) for key in names],[copy.deepcopy(info[output])],[copy.deepcopy(constants[key]) for key in node.input if key in constants]),opset_imports=list(g.opset_import));local.ir_version=g.ir_version
 record={'node':node.name,'operator':node.op_type,'tensor':output,'diagnostic_graph_sha256':hashlib.sha256(local.SerializeToString()).hexdigest(),'input_sha256':{key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in feeds.items()}}
 try:
  expected=ort.InferenceSession(local.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,feeds)[0]
  r=records[output];actual=np.fromfile(a.trace/r['file'],dtype=r['dtype']).reshape(r['shape'])
  record.update(parity=compare(actual,expected,allow_masked_infinity=True),unequal_elements=int(np.count_nonzero(actual!=expected)))
 except Exception as exc:record['error']=str(exc)
 checks.append(record)
artifact=json.loads((a.candidate/'artifacts.json').read_text())
r={'diagnostic_only':True,'candidate':a.candidate.name,'apk_sha256':artifact['files']['app-debug.apk']['sha256'],'model':m['model'],'rtol':1e-3,'atol':1e-4,'checks':checks}
a.output.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'model':m['model'],'local_operator_comparisons':len(checks),'failed':sum(not r.get('parity',{}).get('passed',False) for r in checks),'exact':sum(r.get('unequal_elements')==0 for r in checks),'largest_absolute_errors':sorted([{'operator':r['operator'],'tensor':r['tensor'],'max_abs_error':r.get('parity',{}).get('max_abs_error'),'unequal_elements':r.get('unequal_elements')} for r in checks],key=lambda r:r['max_abs_error'] or 0,reverse=True)[:5]},indent=2))
