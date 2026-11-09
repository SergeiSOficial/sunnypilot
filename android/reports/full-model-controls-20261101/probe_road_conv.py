"""Replay the first road trace violation with its actual GPU input."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'android/app/src/main/python'))
from fold3_model_trace import compare
folder=ROOT/'android/build/candidates/candidate45/traces32/accel-trace-driving_supercombo-buffer'
manifest=json.loads((folder/'manifest.json').read_text())
g=onnx.shape_inference.infer_shapes(onnx.load(ROOT/'android/build/candidates/candidate45/graphs/driving_supercombo-mnn-io.onnx'))
key='android_precision_100_result'
node=next(n for n in g.graph.node if key in n.output)
info={v.name:v for v in (*g.graph.input,*g.graph.value_info,*g.graph.output)}
records={v['tensor']:v for v in manifest['tensors']}
def captured(key):
 t=records[key]
 return np.frombuffer((folder/t['file']).read_bytes(),dtype=t['dtype']).reshape(t['shape'])
inputs={node.input[0]:captured(node.input[0])}
initializers=[copy.deepcopy(v) for v in g.graph.initializer if v.name in node.input[1:]]
local=onnx.helper.make_model(onnx.helper.make_graph([copy.deepcopy(node)],'actual_gpu_input_conv',
 [info[node.input[0]]],[info[key]],initializers),opset_imports=list(g.opset_import))
local.ir_version=g.ir_version
options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
expected=ort.InferenceSession(local.SerializeToString(),options,providers=['CPUExecutionProvider']).run(None,inputs)[0]
actual=captured(key)
result={'candidate':'candidate45','apk_sha256':'9868e5b9ca7d24feaefd9acfbcdabaf4f5226e85063e03fffec4400aecc1a859',
 'diagnostic_only':True,'original_node':node.name,'operator':node.op_type,'tensor':key,
 'input_tensor':node.input[0],'actual_gpu_input_sha256':hashlib.sha256(inputs[node.input[0]].tobytes()).hexdigest(),
 'diagnostic_graph_sha256':hashlib.sha256(local.SerializeToString()).hexdigest(),
 'parity_on_actual_input':compare(actual,expected),
 'note':'An isolated local operation cannot qualify the complete untapped model.'}
out=Path(__file__).with_name('road45-conv10-actual-input.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
