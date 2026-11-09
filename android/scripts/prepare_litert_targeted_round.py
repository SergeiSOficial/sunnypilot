#!/usr/bin/env python3
"""Insert diagnostic half-rounding arithmetic after selected ONNX Gemm nodes."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from decompose_road import digest


def add_rounding(source, names):
  model = onnx.shape_inference.infer_shapes(copy.deepcopy(source))
  types = {v.name: v.type.tensor_type.elem_type for v in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  selected = [n for n in model.graph.node if n.name in names]
  if len(selected) != len(names) or len(set(names)) != len(names):
    raise ValueError('Selected nodes must exist uniquely')
  for node in selected:
    if node.op_type != 'Gemm' or len(node.output) != 1 or types.get(node.output[0]) != T.FLOAT16:
      raise ValueError('Expected a single float16 Gemm output: ' + node.name)
  constants = {'two': 2., 'ten': 10., 'min': -24., 'max': 5., 'ln2': np.log(2.),
               'overflow': 65520., 'inf': np.inf}
  for name, value in constants.items():
    model.graph.initializer.append(nh.from_array(np.array(value, np.float32), 'target_half_' + name))
  nodes = []
  for index, node in enumerate(model.graph.node):
    if node.name not in names:
      nodes.append(node)
      continue
    target = node.output[0]
    node.output[0] = target + '__original_half'
    nodes.append(node)
    prefix = f'target_half_{index}_'
    def op(kind, inputs, suffix, **attrs):
      result = prefix + suffix
      nodes.append(h.make_node(kind, inputs, [result], name=result, **attrs))
      return result
    value = op('Cast', [node.output[0]], 'float', to=T.FLOAT)
    absolute = op('Abs', [value], 'abs')
    logarithm = op('Log', [absolute], 'log')
    exponent = op('Floor', [op('Div', [logarithm, 'target_half_ln2'], 'log2')], 'exponent')
    spacing_exp = op('Clip', [op('Sub', [exponent, 'target_half_ten'], 'exp_minus_ten'),
                              'target_half_min', 'target_half_max'], 'spacing_exp')
    spacing = op('Pow', ['target_half_two', spacing_exp], 'spacing')
    integer = op('Round', [op('Div', [value, spacing], 'scaled')], 'integer')
    rounded = op('Mul', [integer, spacing], 'rounded')
    overflow = op('GreaterOrEqual', [absolute, 'target_half_overflow'], 'overflow')
    infinity = op('Mul', [op('Sign', [value], 'sign'), 'target_half_inf'], 'infinity')
    selected = op('Where', [overflow, infinity, rounded], 'selected')
    nodes.append(h.make_node('Cast', [selected], [target], name=prefix + 'half', to=T.FLOAT16))
  del model.graph.node[:]
  model.graph.node.extend(nodes)
  del model.graph.value_info[:]
  onnx.checker.check_model(model)
  return model


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--source', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--node', action='append', required=True)
  a = p.parse_args()
  if a.output.exists():
    raise ValueError('Preserve existing artifacts')
  onnx.save(add_rounding(onnx.load(a.source), a.node), a.output)
  print(json.dumps({'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
                    'source_sha256': digest(a.source), 'derived_sha256': digest(a.output), 'nodes': a.node}, indent=2))


if __name__ == '__main__':
  main()
