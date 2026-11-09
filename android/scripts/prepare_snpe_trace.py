#!/usr/bin/env python3
"""Extract small road prefixes with original-order flat outputs for DLC tracing."""
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, TensorProto as T
from convert_snpe_linux import digest
from prepare_snpe_sections import required_inputs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'android/app/src/main/python'))
from fold3_sequences import initial_inputs


def prepare(source, output, tensors, seed=42):
  if output.exists(): raise ValueError('Preserve previous trace evidence')
  output.mkdir(parents=True)
  graphs = output/'graphs'; graphs.mkdir()
  fixtures = output/'fixtures'; fixtures.mkdir()
  original = onnx.shape_inference.infer_shapes(onnx.load(source))
  typed = {v.name: v for v in (*original.graph.input,*original.graph.value_info,*original.graph.output)}
  values = initial_inputs(ROOT/'android/app/build/model-reference', 'driving_supercombo', seed)
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'source_sha256': digest(source), 'cases': {}}
  for index, name in enumerate(tensors):
    if name not in typed: raise ValueError('Unknown trace tensor: '+name)
    model = copy.deepcopy(original)
    model.graph.output.append(copy.deepcopy(typed[name]))
    inputs = [v.name for v in model.graph.input]
    model = onnx.utils.Extractor(model).extract_model(inputs, [name])
    if typed[name].type.tensor_type.elem_type != T.FLOAT:
      model.graph.node.append(h.make_node('Cast', [name], ['trace_fp32'], to=T.FLOAT)); name_out='trace_fp32'
    else: name_out=name
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array([-1],np.int64),'trace_shape'))
    model.graph.node.append(h.make_node('Reshape',[name_out,'trace_shape'],['trace_flat']))
    del model.graph.output[:]; model.graph.output.append(h.make_tensor_value_info('trace_flat',T.FLOAT,[None]))
    needed = required_inputs(model)
    keep = [copy.deepcopy(v) for v in model.graph.input if v.name in needed]
    del model.graph.input[:]; model.graph.input.extend(keep)
    model = onnx.shape_inference.infer_shapes(model)
    section='trace_'+str(index); path=graphs/(section+'.onnx'); onnx.save(model,path)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    session=ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])
    folder=fixtures/section;folder.mkdir()
    entry={'section':section,'seed':seed,'original_tensor':name,'section_sha256':digest(path),
           'original_dtype':T.DataType.Name(typed[name].type.tensor_type.elem_type),
           'original_shape':[d.dim_value for d in typed[name].type.tensor_type.shape.dim],
           'inputs':{},'outputs':{}}
    feed={}
    for j,tensor in enumerate(session.get_inputs()):
      dtype={'tensor(float)':np.float32,'tensor(uint8)':np.uint8,'tensor(float16)':np.float16}[tensor.type]
      array=values[tensor.name].reshape(tensor.shape).astype(dtype);feed[tensor.name]=array
      axes=(0,2,3,1) if array.ndim==4 else None
      raw=folder/(str(j)+'.raw');(array.transpose(axes) if axes else array).astype(np.float32).tofile(raw)
      entry['inputs'][tensor.name]={'file':raw.name,'sha256':digest(raw),'axes':axes,'original_shape':list(array.shape),
                                   'dlc_shape':list(array.transpose(axes).shape if axes else array.shape)}
    expected=session.run(None,feed)[0];np.savez(folder/'reference.npz',trace_flat=expected)
    entry['outputs']['trace_flat']={'axes':None,'original_shape':list(expected.shape),'dlc_shape':list(expected.shape)}
    report['cases'][section]=entry
  (fixtures/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
  p.add_argument('--tensor',action='append',required=True)
  p.add_argument('--seed',type=int,default=42)
  args=p.parse_args();prepare(args.source,args.output,args.tensor,args.seed)
