"""Isolate Vulkan linear accumulation using its captured input."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper
import onnxruntime as ort

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(root / 'android/app/src/main/python'))
from fold3_model_trace import compare

folder = Path(__file__).parent / 'ncnn-driver-seed42-reduction'
graph_path = root / 'android/app/build/model-reference/accelerators/dmonitoring_model-mnn-io.onnx'
graph = onnx.shape_inference.infer_shapes(onnx.load(graph_path))
node = next(n for n in graph.graph.node if n.name == 'node_linear')
info = {v.name: v for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
constants = [copy.deepcopy(v) for v in graph.graph.initializer if v.name in node.input]
section = helper.make_model(helper.make_graph([copy.deepcopy(node)], 'same-vulkan-input-gemv',
  [copy.deepcopy(info[node.input[0]])], [copy.deepcopy(info[node.output[0]])], constants),
  opset_imports=list(graph.opset_import), ir_version=graph.ir_version)
onnx.checker.check_model(section)
data = section.SerializeToString()
options = ort.SessionOptions()
options.intra_op_num_threads = 2
actual_input = np.fromfile(folder / 'trace-440.bin', np.float32).astype(np.float16).reshape(1, 1280)
actual_output = np.fromfile(folder / 'trace-441.bin', np.float32).reshape(1, 512)
expected = ort.InferenceSession(data, options, providers=['CPUExecutionProvider']).run(None, {node.input[0]: actual_input})[0]
record = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
  'purpose': 'Compare Vulkan linear against host using the SAME captured Vulkan input',
  'derived_section_sha256': hashlib.sha256(data).hexdigest(),
  'reference_graph_sha256': hashlib.sha256(graph_path.read_bytes()).hexdigest(),
  'input_sha256': hashlib.sha256((folder / 'trace-440.bin').read_bytes()).hexdigest(),
  'output_sha256': hashlib.sha256((folder / 'trace-441.bin').read_bytes()).hexdigest(),
  'parity': compare(actual_output, expected)}
(Path(__file__).parent / 'ncnn-same-input-gemv.json').write_text(json.dumps(record, indent=2) + '\n')
print(record['parity'])
