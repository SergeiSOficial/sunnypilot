"""Check captured MNN Pow outputs on their actual GPU inputs, without upstream drift."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'android/app/src/main/python'))
from fold3_model_trace import compare
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=json.loads((a.trace/'manifest.json').read_text());g=onnx.shape_inference.infer_shapes(onnx.load(a.candidate/'graphs'/(m['model']+'-mnn-io.onnx')))
records={r['tensor']:r for r in m['tensors']};info={v.name:v for v in (*g.graph.input,*g.graph.value_info,*g.graph.output)};const={v.name:v for v in g.graph.initializer};checks=[]
def read(key):
 r=records[key];return np.frombuffer((a.trace/r['file']).read_bytes(),dtype=r['dtype']).reshape(r['shape'])
for node in g.graph.node:
 if node.op_type!='Pow' or node.output[0] not in records or node.input[0] not in records or node.input[1] not in const:continue
 x=read(node.input[0]);local=onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(node)],'actual_gpu_input_power',[info[node.input[0]]],[info[node.output[0]]],[copy.deepcopy(const[node.input[1]])]),opset_imports=list(g.opset_import));local.ir_version=g.ir_version
 options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
 expected=ort.InferenceSession(local.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,{node.input[0]:x})[0];actual=read(node.output[0]);exponent=onnx.numpy_helper.to_array(const[node.input[1]])
 mul=np.multiply(np.multiply(x,x),x) if exponent.size==1 and float(exponent.reshape(-1)[0])==3 else None
 checks.append({'tensor':node.output[0],'input':node.input[0],'shape':list(x.shape),'exponent':exponent.tolist(),'diagnostic_graph_sha256':hashlib.sha256(local.SerializeToString()).hexdigest(),'gpu_input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'gpu_on_actual_input':compare(actual,expected),'gpu_unequal_elements':int(np.count_nonzero(actual!=expected)),'separate_mul_host_parity':None if mul is None else compare(mul,expected),'separate_mul_unequal_elements':None if mul is None else int(np.count_nonzero(mul!=expected))})
r={'diagnostic_only':True,'candidate':a.candidate.name,'model':m['model'],'apk_sha256':json.loads((a.candidate/'artifacts.json').read_text())['files']['app-debug.apk']['sha256'],'checks':checks};a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
