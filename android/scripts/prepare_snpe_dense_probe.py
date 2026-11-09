#!/usr/bin/env python3
"""Isolate the first road dense operation on identical, original half inputs."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper as h, TensorProto as T
from convert_snpe_linux import digest


def ordered_half_gemv(x, weight, bias):
  """Diagnostic of the pinned ARM half GEMV's eight half-FMA lanes."""
  if x.ndim != 2 or x.shape[0] != 1 or weight.ndim != 2 or weight.shape[1] != x.shape[1] or x.shape[1] % 8:
    raise ValueError('Unsupported ordered-half GEMV shape')
  if bias.shape != (weight.shape[0],): raise ValueError('Unsupported bias shape')
  x = x.astype(np.float16).astype(np.float64)
  weight = weight.astype(np.float16).astype(np.float64)
  lanes = np.zeros((weight.shape[0], 8), np.float16)
  for start in range(0, x.shape[1], 8):
    lanes = (lanes.astype(np.float64) + weight[:,start:start+8]*x[:,start:start+8]).astype(np.float16)
  for _ in range(3):
    lanes = (lanes[:,::2].astype(np.float64)+lanes[:,1::2].astype(np.float64)).astype(np.float16)
  return (lanes[:,0].astype(np.float64)+bias.astype(np.float16).astype(np.float64)).astype(np.float16).reshape(1,-1)


def prepare(source, mean_reference, linear_reference, output):
  if output.exists(): raise ValueError('Preserve earlier probe evidence')
  model = onnx.load(source)
  node = copy.deepcopy(next(n for n in model.graph.node if list(n.output) == ['model/linear']))
  initializers = {w.name: w for w in model.graph.initializer}
  weights = [copy.deepcopy(initializers[name]) for name in node.input[1:]]
  if node.op_type != 'Gemm' or len(weights) != 2 or any(w.data_type != T.FLOAT16 for w in weights):
    raise ValueError('First dense contract changed')
  attributes = {a.name: h.get_attribute_value(a) for a in node.attribute}
  if attributes != {'alpha': 1., 'beta': 1., 'transA': 0, 'transB': 1}:
    raise ValueError('First dense arithmetic changed')
  with np.load(mean_reference) as reference:
    x = reference['trace_flat'].astype(np.float16).reshape(1, -1)
  if x.shape[1] != weights[0].dims[1]: raise ValueError('Mean reference shape changed')
  node.input[0] = 'half_input'; node.output[0] = 'half_output'
  nodes = [h.make_node('Cast', ['input'], ['half_input'], to=T.FLOAT16), node,
           h.make_node('Cast', ['half_output'], ['trace_flat'], to=T.FLOAT)]
  graph = h.make_model(h.make_graph(nodes, 'dense',
    [h.make_tensor_value_info('input', T.FLOAT, list(x.shape))],
    [h.make_tensor_value_info('trace_flat', T.FLOAT, [1, weights[0].dims[0]])], weights),
    opset_imports=[h.make_opsetid('', 20)], ir_version=10)
  onnx.checker.check_model(graph)
  graphs = output/'graphs'; graphs.mkdir(parents=True)
  path = graphs/'dense.onnx'; onnx.save(graph, path)
  feed = {'input': x.astype(np.float32)}
  with np.load(linear_reference) as captured:
    expected = captured['trace_flat'].astype(np.float32).reshape(1,-1)
  ordered = ordered_half_gemv(x, onnx.numpy_helper.to_array(weights[0]), onnx.numpy_helper.to_array(weights[1]))
  if not np.array_equal(ordered, expected): raise ValueError('Captured dense reference does not match ordered half inputs')
  host = ort.InferenceSession(str(path), providers=['CPUExecutionProvider']).run(None, feed)[0]
  # A separate widened graph diagnoses whether a converter erased half rounding.
  widened = copy.deepcopy(graph)
  for weight in widened.graph.initializer:
    weight.CopyFrom(onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(weight).astype(np.float32), weight.name))
  widened.graph.node[0].attribute[0].i = T.FLOAT
  fp32 = ort.InferenceSession(widened.SerializeToString(), providers=['CPUExecutionProvider']).run(None, feed)[0]
  folder = output/'fixtures'/'dense'; folder.mkdir(parents=True)
  raw = folder/'input.raw'; feed['input'].tofile(raw)
  reference = folder/'reference.npz'; np.savez(reference, trace_flat=expected, diagnostic_fp32=fp32,
                                              diagnostic_ordered_half=ordered, diagnostic_host=host)
  case = {'section': 'dense', 'section_sha256': digest(path), 'original_node': node.name,
    'inputs': {'input': {'file': raw.name, 'sha256': digest(raw), 'axes': None,
                         'original_shape': list(x.shape), 'dlc_shape': list(x.shape)}},
    'outputs': {'trace_flat': {'axes': None, 'original_shape': list(expected.shape), 'dlc_shape': list(expected.shape)}}}
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'source_sha256': digest(source),
    'mean_reference_sha256': digest(mean_reference), 'reference_sha256': digest(reference), 'cases': {'dense': case}}
  report['linear_reference_sha256'] = digest(linear_reference)
  report['ordered_half_matches_capture_exactly'] = True
  report['local_host_matches_capture'] = bool(np.allclose(host, expected, rtol=1e-3, atol=1e-4))
  (output/'fixtures'/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('source', 'mean-reference', 'linear-reference', 'output'): p.add_argument('--'+name, type=Path, required=True)
  args = p.parse_args(); prepare(args.source, args.mean_reference, args.linear_reference, args.output)
