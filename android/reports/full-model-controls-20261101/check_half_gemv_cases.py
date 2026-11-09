"""Check the pinned reference's half GEMV tails/scalar/coefficient reductions."""
import json,sys
from pathlib import Path
import numpy as np,onnx,onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'app/src/main/python'))
from fold3_model_trace import compare
records=[]
for k,n in ((1280,512),(515,96),(515,128),(32,1),(16,1),(16,2),(8,2),(2,512),(19,32),(64,1),(24,1)):
 adef=onnx.helper.make_tensor_value_info('a',10,[1,k]);bdef=onnx.helper.make_tensor_value_info('b',10,[n,k]);cdef=onnx.helper.make_tensor_value_info('c',10,[n]);out=onnx.helper.make_tensor_value_info('out',10,[1,n])
 graph=onnx.helper.make_model(onnx.helper.make_graph([onnx.helper.make_node('Gemm',['a','b','c'],['out'],transB=1)],'gemv',[adef,bdef,cdef],[out]),opset_imports=[onnx.helper.make_opsetid('',21)]);graph.ir_version=10
 options=ort.SessionOptions();options.intra_op_num_threads=2
 session=ort.InferenceSession(graph.SerializeToString(),options,providers=['CPUExecutionProvider'])
 for seed in (42,7,123):
  rng=np.random.default_rng(seed);a=rng.normal(size=(1,k)).astype(np.float16);b=rng.normal(size=(n,k)).astype(np.float16);c=rng.normal(size=n).astype(np.float16)
  end=k//8*8;coefficient=end>0 and k+n+1<20
  packets=min(4,end//8) if n==1 and not coefficient and end else 1
  lanes=np.zeros((n,packets,8),np.float16)
  if coefficient:
   lanes[:,0]= (b[:,:8].astype(np.float32)*a[0,:8].astype(np.float32)).astype(np.float16)
   if end>8:
    other=(b[:,8:16].astype(np.float32)*a[0,8:16].astype(np.float32)).astype(np.float16)
    lanes[:,0]=(lanes[:,0].astype(np.float32)+other.astype(np.float32)).astype(np.float16)
  else:
   for start in range(0,end,8):
    packet=(start//8)%packets
    lanes[:,packet]=(b[:,start:start+8].astype(np.float64)*a[0,start:start+8].astype(np.float64)+lanes[:,packet].astype(np.float64)).astype(np.float16)
  partial=lanes[:,packets-1]
  for packet in range(packets-2,-1,-1):partial=(lanes[:,packet].astype(np.float32)+partial.astype(np.float32)).astype(np.float16)
  for width in (4,2,1):partial=(partial[:,::2].astype(np.float32)+partial[:,1::2].astype(np.float32)).astype(np.float16).reshape(n,width)
  value=partial[:,0]
  for i in range(end,k):value=(value.astype(np.float32)+(b[:,i].astype(np.float32)*a[0,i].astype(np.float32)).astype(np.float16).astype(np.float32)).astype(np.float16)
  actual=(value.astype(np.float32)+c.astype(np.float32)).astype(np.float16).reshape(1,n)
  expected=session.run(None,{'a':a,'b':b,'c':c})[0]
  records.append({'k':k,'n':n,'seed':seed,**compare(actual,expected)})
report={'passed':all(r['passed'] for r in records),'checks':records}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print('half GEMV:',sum(r['passed'] for r in records),'/',len(records),'passed')
if not report['passed']:
 print([r for r in records if not r['passed']]);raise SystemExit(1)
