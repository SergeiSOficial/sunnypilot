"""Diagnostic FP32 transport with explicit original FP16 rounding boundaries.

MNN and pnnx discard ONNX Cast(FLOAT16). These graphs instead express each
half result as an exactly representable float, keeping external adapters and
original-reference qualification mandatory. They are never implicit fallbacks.
"""
import copy

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def explicit_half(model):
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
  types = {v.name: v.type.tensor_type.elem_type for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  constants = {"two": 2., "ten": 10., "min": -24., "max": 5., "ln2": np.log(2.), "overflow": 65536., "inf": np.inf}
  for name, value in constants.items():
    model.graph.initializer.append(nh.from_array(np.array(value, np.float32), "fold3_half_" + name))
  for index, tensor in enumerate(model.graph.initializer):
    if tensor.data_type in (T.FLOAT16, T.UINT8):
      model.graph.initializer[index].CopyFrom(nh.from_array(nh.to_array(tensor).astype(np.float32), tensor.name))
  for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
    if value.type.tensor_type.elem_type in (T.FLOAT16, T.UINT8):
      value.type.tensor_type.elem_type = T.FLOAT
  nodes = []
  moves = {"Reshape", "Transpose", "Concat", "Split", "Unsqueeze", "Squeeze", "Slice", "Gather", "GatherND", "Expand", "Identity"}
  for index, node in enumerate(model.graph.node):
    if node.op_type == "Cast":
      for attribute in node.attribute:
        if attribute.name == "to" and attribute.i in (T.FLOAT16, T.UINT8):
          attribute.i = T.FLOAT
    boundary = [i for i, name in enumerate(node.output) if types.get(name) == T.FLOAT16] if node.op_type not in moves else []
    for output_index in boundary:
      node.output[output_index] += "__before_half"
    nodes.append(node)
    for output_index in boundary:
      source = node.output[output_index]
      target = source.removesuffix("__before_half")
      prefix = f"fold3_round_{index}_{output_index}_"
      def op(kind, inputs, output, **attributes):
        result = prefix + output
        nodes.append(h.make_node(kind, inputs, [result], name=result, **attributes))
        return result
      absolute = op("Abs", [source], "abs")
      logarithm = op("Log", [absolute], "log")
      exponent = op("Floor", [op("Div", [logarithm, "fold3_half_ln2"], "log2")], "exponent")
      spacing_exponent = op("Sub", [exponent, "fold3_half_ten"], "spacing_exponent")
      clipped = op("Clip", [spacing_exponent, "fold3_half_min", "fold3_half_max"], "clipped")
      spacing = op("Pow", ["fold3_half_two", clipped], "spacing")
      scaled = op("Div", [source, spacing], "scaled")
      integer = op("Round", [scaled], "integer")
      rounded = op("Mul", [integer, spacing], "rounded")
      overflow = op("GreaterOrEqual", [op("Abs", [rounded], "rounded_abs"), "fold3_half_overflow"], "overflow")
      infinity = op("Mul", [op("Sign", [source], "sign"), "fold3_half_inf"], "infinity")
      nodes.append(h.make_node("Where", [overflow, infinity, rounded], [target], name=prefix + "select"))
  del model.graph.node[:]
  model.graph.node.extend(nodes)
  del model.graph.value_info[:]
  onnx.checker.check_model(model)
  return model
