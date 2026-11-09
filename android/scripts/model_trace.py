#!/usr/bin/env python3
"""Export bounded, observable operator traces without replacing original goldens."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]


def infer_trace_types(model):
  """Seed ORT custom-op types before propagating standard ONNX types."""
  model = onnx.shape_inference.infer_shapes(model)
  types = {value.name: value.type.tensor_type.elem_type
           for value in (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  types.update({value.name: value.data_type for value in model.graph.initializer})
  for node in model.graph.node:
    if node.domain != "com.microsoft":
      continue
    source = 1 if node.op_type in ("FusedConv", "NhwcFusedConv", "FusedGemm") else 0 if node.op_type == "QuickGelu" else None
    if source is None or not types.get(node.input[source]):
      continue
    for output in node.output:
      if output and not types.get(output):
        types[output] = types[node.input[source]]
        model.graph.value_info.append(onnx.helper.make_tensor_value_info(output, types[output], None))
  return onnx.shape_inference.infer_shapes(model)


def promote_nodes(model, names):
  """Diagnostic FP32 islands; retain the original FP16 tensor boundaries."""
  types = {v.name: v.type.tensor_type.elem_type for v in
           (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  types.update({v.name: v.data_type for v in model.graph.initializer})
  missing = set(names) - {node.name for node in model.graph.node}
  if missing:
    raise ValueError(f"Unknown precision nodes: {sorted(missing)}")
  nodes = []
  for node in model.graph.node:
    if node.name not in names:
      nodes.append(node)
      continue
    if node.domain or node.op_type not in ("Conv", "Gelu", "MatMul", "Gemm"):
      raise ValueError(f"Unsupported precision island: {node.name}")
    for i, value in enumerate(node.input):
      if value and types.get(value) == onnx.TensorProto.FLOAT16:
        converted = f"{node.name}_fp32_input_{i}"
        nodes.append(onnx.helper.make_node("Cast", [value], [converted], to=onnx.TensorProto.FLOAT))
        node.input[i] = converted
    casts = []
    for i, value in enumerate(node.output):
      if types.get(value) != onnx.TensorProto.FLOAT16:
        raise ValueError(f"Precision island requires declared FP16 outputs: {value}")
      converted = f"{node.name}_fp32_output_{i}"
      node.output[i] = converted
      casts.append(onnx.helper.make_node("Cast", [converted], [value], to=onnx.TensorProto.FLOAT16))
    nodes.extend((node, *casts))
  del model.graph.node[:]
  model.graph.node.extend(nodes)
  del model.graph.value_info[:]
  return onnx.shape_inference.infer_shapes(model)


def float_image_ops(model):
  """Lift lossless uint8 image rearrangement into QNN-supported FP16 operators.

  All 256 byte values are exactly representable in FP16. External image/state
  types stay uint8; only index/shape operators move across their existing cast.
  Never apply this to arithmetic, quantization, or index tensors.
  """
  model = onnx.shape_inference.infer_shapes(model)
  types = {v.name: v.type.tensor_type.elem_type for v in
           (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  types.update({v.name: v.data_type for v in model.graph.initializer})
  external = {v.name for v in model.graph.output}
  rearrange = {"Slice", "Concat", "Gather", "Split", "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Identity"}
  lifted, nodes = {}, []
  for node in model.graph.node:
    eligible = not node.domain and node.op_type in rearrange and all(
      types.get(value) == onnx.TensorProto.UINT8 for value in node.output if value)
    for i, value in enumerate(node.input):
      if not value or types.get(value) != onnx.TensorProto.UINT8:
        continue
      if eligible and (i == 0 or node.op_type == "Concat"):
        if value not in lifted:
          lifted[value] = value + "_android_image_half"
          nodes.append(onnx.helper.make_node("Cast", [value], [lifted[value]], to=onnx.TensorProto.FLOAT16))
        node.input[i] = lifted[value]
      elif node.op_type == "Cast" and value in lifted:
        node.input[i] = lifted[value]
    casts = []
    if eligible:
      for i, value in enumerate(node.output):
        if value:
          lifted[value] = value + "_android_image_half"
          node.output[i] = lifted[value]
          # Preserve original uint8 graph outputs and any unmodified consumers.
          casts.append(onnx.helper.make_node("Cast", [lifted[value]], [value], to=onnx.TensorProto.UINT8))
    nodes.extend((node, *casts))
  # Unused cast-back nodes are removed by the provider optimizer. Keep external
  # outputs even when they have no downstream consumer.
  used = external | {value for node in nodes for value in node.input}
  nodes = [node for node in nodes if any(value in used for value in node.output)]
  del model.graph.node[:]
  model.graph.node.extend(nodes)
  del model.graph.value_info[:]
  return lower_image_ranks(onnx.shape_inference.infer_shapes(model))


def lower_image_ranks(model):
  """Fold H/W for five-axis image Slice/Concat; preserve all external axes.

  QNN GPU rejects the road history's rank-five Slice and Concat. These index
  only the leading history axes, so folding the untouched trailing H/W axes
  is an exact reshape, with no change to byte values or history ordering.
  """
  shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in
            (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  types = {v.name: v.type.tensor_type.elem_type for v in
           (*model.graph.input, *model.graph.value_info, *model.graph.output)}
  constants = {v.name: onnx.numpy_helper.to_array(v) for v in model.graph.initializer}
  nodes = []
  for ordinal, node in enumerate(model.graph.node):
    source = shapes.get(node.input[0], []) if node.input else []
    axes = ([next((a.i for a in node.attribute if a.name == "axis"), 0)] if node.op_type == "Concat" else
            constants.get(node.input[3], []) if node.op_type == "Slice" and len(node.input) > 3 else [])
    if (node.domain or node.op_type not in ("Slice", "Concat") or len(source) != 5 or not all(source)
        or types.get(node.input[0]) != onnx.TensorProto.FLOAT16 or not len(axes)
        or any(int(axis) % 5 >= 3 for axis in axes)):
      nodes.append(node)
      continue
    prefix = f"android_image_rank_{ordinal}"
    for i in range(len(node.input) if node.op_type == "Concat" else 1):
      value = node.input[i]
      shape = shapes[value]
      packed = [*shape[:3], shape[3] * shape[4]]
      shape_name, tensor_name = f"{prefix}_shape_{i}", f"{prefix}_input_{i}"
      model.graph.initializer.append(onnx.numpy_helper.from_array(np.array(packed, np.int64), shape_name))
      nodes.append(onnx.helper.make_node("Reshape", [value, shape_name], [tensor_name]))
      node.input[i] = tensor_name
    output = node.output[0]
    node.output[0] = prefix + "_output"
    # Normalize negative leading axes before rank reduction.
    if node.op_type == "Concat":
      next(a for a in node.attribute if a.name == "axis").i = int(axes[0]) % 5
    elif any(int(axis) < 0 for axis in axes):
      axis_name = prefix + "_axes"
      model.graph.initializer.append(onnx.numpy_helper.from_array(np.array(axes, np.int64) % 5, axis_name))
      node.input[3] = axis_name
    shape_name = prefix + "_output_shape"
    model.graph.initializer.append(onnx.numpy_helper.from_array(np.array(shapes[output], np.int64), shape_name))
    nodes.extend((node, onnx.helper.make_node("Reshape", [node.output[0], shape_name], [output])))
  del model.graph.node[:]
  model.graph.node.extend(nodes)
  del model.graph.value_info[:]
  return onnx.shape_inference.infer_shapes(model)


def observable_model(source, destination, start, limit, optimized=False, promote=(), portable_host=False, float_images=False):
  """Observe the already optimized graph so output taps cannot undo fusion."""
  original = onnx.shape_inference.infer_shapes(onnx.load(source))
  if portable_host:
    from reference_precision import portable
    frozen = ROOT / f"android/app/build/model-reference/host-fusion/{source.name}"
    original = onnx.shape_inference.infer_shapes(portable(onnx.load(frozen), promote_half_gemm=False))
  if promote:
    if optimized:
      raise ValueError("Precision islands require the original graph")
    original = promote_nodes(original, promote)
  if float_images:
    original = float_image_ops(original)
  values = {value.name: value for value in (*original.graph.input, *original.graph.value_info, *original.graph.output)}
  if optimized:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.optimized_model_filepath = str(destination / (source.stem + "-host.onnx"))
    ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
    model = onnx.load(options.optimized_model_filepath)
    # ORT adds imports for unused experimental domains. Keep only domains used
    # by actual nodes so the pinned ONNX parser need not accept unknown domains.
    used = {node.domain for node in model.graph.node} | {""}
    imports = [value for value in model.opset_import if value.domain in used]
    del model.opset_import[:]
    model.opset_import.extend(imports)
    onnx.save(model, options.optimized_model_filepath)
    # Optimizer-created float32 precision boundaries are absent from the
    # original graph's value_info. Include their inferred types in the trace.
    inferred = infer_trace_types(model)
    values.update({value.name: value for value in (*inferred.graph.value_info, *inferred.graph.output)})
  else:
    model = original
  originals = {value.name for value in model.graph.output}
  nodes = []
  for index, node in enumerate(model.graph.node[start:start + limit], start=start):
    for output in node.output:
      if output and output not in originals and output in values:
        value = values[output]
        # ORT's internal NHWC transforms retain names but change shapes.
        model.graph.output.append(onnx.helper.make_tensor_value_info(output, value.type.tensor_type.elem_type, None)
                                  if optimized else value)
        originals.add(output)
        tensor = value.type.tensor_type
        nodes.append({"index": index, "operator": node.op_type, "name": node.name, "output": output,
                      "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                      "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param for dim in tensor.shape.dim],
                      "inputs": list(node.input), "domain": node.domain,
                      "layout": "ONNX declared axes; provider transformations recorded separately"})
  return model, nodes


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--limit", type=int, default=40)
  parser.add_argument("--start", type=int, default=0)
  parser.add_argument("--directory", choices=("trace", "trace-tail", "trace-fused"), default="trace")
  parser.add_argument("--output-directory", type=Path,
                      help="Isolated diagnostic export; does not update packaged traces or host-fusion models")
  parser.add_argument("--optimized", action="store_true", help="Freeze host fusion before adding tensor outputs")
  parser.add_argument("--promote-node", action="append", default=[], help="Diagnostic FP32 island with original FP16 boundaries")
  parser.add_argument("--portable-host", action="store_true", help="Diagnostic portable host precision boundaries; retain half Gemm")
  parser.add_argument("--float-image-ops", action="store_true", help="Diagnostic lossless float image rearrangement for QNN GPU")
  parser.add_argument("--model", choices=("both", "driving_supercombo", "dmonitoring_model"), default="both")
  args = parser.parse_args()
  if not 1 <= args.limit <= 1000:
    parser.error("limit must be between 1 and 1000 operators")
  if not 0 <= args.start < 1000:
    parser.error("start must be between 0 and 999")
  references = ROOT / "android/app/build/model-reference"
  destination = args.output_directory.resolve() if args.output_directory else references / args.directory
  if args.output_directory and destination.is_relative_to(references.resolve()):
    parser.error("isolated output-directory must be outside the packaged model-reference directory")
  destination.mkdir(parents=True, exist_ok=True)
  manifest = {"scope": "diagnostic: frozen host fusion" if args.optimized else
              "diagnostic: observing intermediate tensors can change fusion/rounding", "optimized": args.optimized, "models": {}}
  fusion = {"scope": "Frozen host precision/fusion boundaries; original-reference qualification required", "models": {}}
  for name in (("dmonitoring_model", "driving_supercombo") if args.model == "both" else (args.model,)):
    source = ROOT / "openpilot/selfdrive/modeld/models" / f"{name}.onnx"
    model, nodes = observable_model(source, destination, args.start, args.limit, args.optimized, args.promote_node,
                                    args.portable_host, args.float_image_ops)
    path = destination / f"{name}.onnx"
    onnx.save(model, path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    if args.optimized:
      options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    with np.load(references / f"{name}-inputs.npz") as inputs:
      outputs = dict(zip((spec.name for spec in session.get_outputs()), session.run(None, dict(inputs)), strict=True))
    np.savez_compressed(destination / f"{name}-outputs.npz", **outputs)
    with np.load(references / f"{name}-outputs.npz") as frozen:
      original_checks = {key: bool(np.allclose(outputs[key], frozen[key], rtol=1e-3, atol=1e-4)) for key in frozen.files}
    if args.optimized and not all(original_checks.values()):
      raise RuntimeError(f"Frozen-fusion trace changes original outputs: {name}: {original_checks}")
    manifest["models"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
      "original_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "nodes": nodes, "promoted_nodes": args.promote_node,
      "portable_host": args.portable_host,
      "float_image_ops": args.float_image_ops,
      "instrumented_matches_original": original_checks}
    if args.optimized and not args.output_directory:
      folder = references / "host-fusion"
      folder.mkdir(exist_ok=True)
      compiled = folder / f"{name}.onnx"
      shutil.copyfile(destination / f"{name}-host.onnx", compiled)
      fusion["models"][name] = {"sha256": hashlib.sha256(compiled.read_bytes()).hexdigest(),
                                "original_sha256": manifest["models"][name]["original_sha256"]}
  if args.optimized and not args.output_directory:
    (references / "host-fusion/manifest.json").write_text(json.dumps(fusion, indent=2) + "\n")
  (destination / "manifest.json").write_text(json.dumps(manifest, indent=2))
  print(json.dumps({name: {"operators": len(spec["nodes"]), "instrumented_matches_original": spec["instrumented_matches_original"]}
                    for name, spec in manifest["models"].items()}, indent=2))


if __name__ == "__main__":
  main()
