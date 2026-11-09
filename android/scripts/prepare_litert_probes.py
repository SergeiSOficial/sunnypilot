#!/usr/bin/env python3
"""Lower original Conv/Cast primitives without changing numerical dtype contracts."""
import argparse
import json
from pathlib import Path
import hashlib
import flatbuffers
import numpy as np
import onnx
from onnx import numpy_helper
from ai_edge_litert import schema_py_generated as s
from ai_edge_litert.interpreter import Interpreter


def lower(source, explicit_half_conv=False, float_transport=False):
  original = onnx.load(source); original = onnx.shape_inference.infer_shapes(original)
  shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
            for v in (*original.graph.input,*original.graph.value_info,*original.graph.output)}
  model=s.ModelT();model.version=3;model.buffers=[s.BufferT()];model.operatorCodes=[]
  graph=s.SubGraphT();graph.name='original_probe';graph.tensors=[];graph.operators=[];model.subgraphs=[graph]
  ids={}
  def tensor(name,shape,dtype,array=None):
    t=s.TensorT();t.name=name;t.shape=np.array(shape,np.int32);t.type=dtype;t.hasRank=True
    if array is not None:
      buffer=s.BufferT();buffer.data=np.frombuffer(array.tobytes(),np.uint8);t.buffer=len(model.buffers);model.buffers.append(buffer)
    ids[name]=len(graph.tensors);graph.tensors.append(t);return ids[name]
  def op(code,inputs,outputs,options=None):
    c=s.OperatorCodeT();c.builtinCode=code;c.deprecatedBuiltinCode=min(code,127);c.version=1
    index=len(model.operatorCodes);model.operatorCodes.append(c)
    n=s.OperatorT();n.opcodeIndex=index;n.inputs=np.array(inputs,np.int32);n.outputs=np.array(outputs,np.int32)
    if options is not None:n.builtinOptionsType=getattr(s.BuiltinOptions,type(options).__name__[:-1]);n.builtinOptions=options
    graph.operators.append(n)
  dtype={onnx.TensorProto.FLOAT:s.TensorType.FLOAT32,onnx.TensorProto.FLOAT16:s.TensorType.FLOAT16}
  inp=original.graph.input[0];tensor(inp.name,shapes[inp.name],dtype[inp.type.tensor_type.elem_type])
  if all(n.op_type=='Cast' for n in original.graph.node):
    for node in original.graph.node:
      ty=dtype[next(a.i for a in node.attribute if a.name=='to')]
      output=tensor(node.output[0],shapes[node.output[0]],ty)
      option=s.CastOptionsT();option.inDataType=graph.tensors[ids[node.input[0]]].type;option.outDataType=ty
      op(s.BuiltinOperator.CAST,[ids[node.input[0]]],[output],option)
  elif len(original.graph.node)==1 and original.graph.node[0].op_type=='Conv':
    node=original.graph.node[0];weights={w.name:numpy_helper.to_array(w) for w in original.graph.initializer}
    attributes = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
    allowed = {'kernel_shape': [3,3], 'strides': [1,1], 'pads': [0,0,0,0], 'dilations': [1,1], 'group': 1, 'auto_pad': b'NOTSET'}
    if any(name not in allowed or value != allowed[name] for name,value in attributes.items()):
      raise ValueError('Unsupported Conv geometry; refusing an approximate conversion')
    ty=graph.tensors[0].type
    input_id = 0
    if float_transport and ty == s.TensorType.FLOAT16:
      if not explicit_half_conv: raise ValueError('Float transport requires explicit half Conv')
      graph.tensors[0].type = s.TensorType.FLOAT32
      input_id = tensor('input_fp16', shapes[inp.name], ty)
      cast=s.CastOptionsT();cast.inDataType=s.TensorType.FLOAT32;cast.outDataType=ty
      op(s.BuiltinOperator.CAST,[0],[input_id],cast)
    # Explicit transposes retain the ONNX external NCHW contract.
    perm=tensor('to_nhwc_perm',[4],s.TensorType.INT32,np.array([0,2,3,1],np.int32))
    n,c,h,w=shapes[inp.name];nhwc=tensor('nhwc',[n,h,w,c],ty)
    op(s.BuiltinOperator.TRANSPOSE,[input_id,perm],[nhwc],s.TransposeOptionsT())
    output_type = ty
    if ty == s.TensorType.FLOAT16 and explicit_half_conv:
      widened=tensor('conv_input_fp32',[n,h,w,c],s.TensorType.FLOAT32)
      cast=s.CastOptionsT();cast.inDataType=ty;cast.outDataType=s.TensorType.FLOAT32
      op(s.BuiltinOperator.CAST,[nhwc],[widened],cast);nhwc=widened;ty=s.TensorType.FLOAT32
    numpy_type=np.float32 if ty==s.TensorType.FLOAT32 else np.float16
    kernel=weights[node.input[1]].transpose(0,2,3,1).astype(numpy_type);bias=weights[node.input[2]].astype(numpy_type)
    wi=tensor('weights',kernel.shape,ty,kernel);bi=tensor('bias',bias.shape,ty,bias)
    n,c,h,w=shapes[node.output[0]];conv=tensor('conv_nhwc',[n,h,w,c],ty)
    opts=s.Conv2DOptionsT();opts.padding=s.Padding.VALID;opts.strideW=opts.strideH=1
    op(s.BuiltinOperator.CONV_2D,[nhwc,wi,bi],[conv],opts)
    if ty != output_type:
      rounded=tensor('conv_output_fp16',[n,h,w,c],output_type)
      cast=s.CastOptionsT();cast.inDataType=ty;cast.outDataType=output_type
      op(s.BuiltinOperator.CAST,[conv],[rounded],cast);conv=rounded
    perm=tensor('to_nchw_perm',[4],s.TensorType.INT32,np.array([0,3,1,2],np.int32))
    out=tensor(node.output[0],shapes[node.output[0]],output_type)
    op(s.BuiltinOperator.TRANSPOSE,[conv,perm],[out],s.TransposeOptionsT())
    if float_transport and output_type == s.TensorType.FLOAT16:
      widened=tensor(node.output[0]+'/float_transport',shapes[node.output[0]],s.TensorType.FLOAT32)
      cast=s.CastOptionsT();cast.inDataType=output_type;cast.outDataType=s.TensorType.FLOAT32
      op(s.BuiltinOperator.CAST,[out],[widened],cast)
      ids[node.output[0]]=widened
  else:raise ValueError('Unimplemented lowering; no graph approximation allowed')
  graph.inputs=np.array([0],np.int32);graph.outputs=np.array([ids[original.graph.output[0].name]],np.int32)
  builder=flatbuffers.Builder(1024);offset=model.Pack(builder);builder.Finish(offset,file_identifier=b'TFL3')
  return bytes(builder.Output())


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
  p.add_argument('--explicit-half-conv',action='store_true',help='Diagnostic FP32 Conv with original FP16 IO casts')
  p.add_argument('--float-transport',action='store_true',help='Lossless FP32 external transport; retain explicit internal half casts')
  args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);report={'qualified':False,'vehicle_tx_enabled':False,'probes':{}}
  for name in ('fp32-conv','fp16-conv','fp32-cast'):
    source=args.probes/(name+'.onnx');data=lower(source,args.explicit_half_conv,args.float_transport);path=args.output/(name+'.tflite');path.write_bytes(data)
    graph=s.Model.GetRootAsModel(data,0).Subgraphs(0)
    types={s.TensorType.FLOAT16:'float16',s.TensorType.FLOAT32:'float32'}
    entry={'explicit_half_conv':args.explicit_half_conv,'float_transport':args.float_transport,
           'input_dtype':types[graph.Tensors(graph.Inputs(0)).Type()],
           'output_dtype':types[graph.Tensors(graph.Outputs(0)).Type()],
           'onnx_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'tflite_sha256':hashlib.sha256(data).hexdigest()}
    try:
      interpreter=Interpreter(model_path=str(path),num_threads=1,experimental_delegates=[]);interpreter.allocate_tensors()
      with np.load(args.probes/(name+'.npz')) as fixture:
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'],fixture['x'].astype(entry['input_dtype']));interpreter.invoke()
        actual=interpreter.get_tensor(interpreter.get_output_details()[0]['index']);expected=fixture['expected']
        entry.update(host_cpu_parity=bool(np.allclose(actual,expected,rtol=1e-3,atol=1e-4)),max_abs_error=float(np.abs(actual-expected).max()))
    except Exception as exc:entry['host_cpu_error']=str(exc)
    report['probes'][name]=entry;print(name,entry,flush=True)
  (args.output/'conversion.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
