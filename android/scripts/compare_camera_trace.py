#!/usr/bin/env python3
"""Locate device replay divergence against the frozen, original host graph."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_model_trace import compare, compare_replay_frame
from fold3_sequences import digest, initial_inputs
from model_trace import infer_trace_types


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--archive", type=Path, required=True)
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--model", choices=("dmonitoring_model", "driving_supercombo"), default="dmonitoring_model")
  parser.add_argument("--recurrent", action="store_true", help="Trace road seed 7, recurrent frame 10 using archived device feedback")
  args = parser.parse_args()
  references = ROOT / "android/app/build/model-reference"
  name = args.model
  road = name == "driving_supercombo"
  if args.recurrent and not road:
    parser.error("--recurrent requires --model driving_supercombo")
  frame = 10 if args.recurrent else 0 if road else 41
  seed = 7 if args.recurrent else 42
  label = f"seed{seed}" if args.recurrent else "camera-replay"
  source = references / f"host-fusion/{name}.onnx"
  metadata = json.loads((source.parent / "manifest.json").read_text())["models"][name]
  if digest(source) != metadata["sha256"]:
    raise ValueError("Frozen host graph hash mismatch")
  manifest = json.loads((references / "sequences/manifest.json").read_text())
  inputs = initial_inputs(references, name, seed)
  input_lineage = None
  if args.recurrent:
    from fold3_sequences import output_record
    with np.load(args.archive) as archive, np.load(references / f"sequences/{name}-{label}.npz") as golden:
      if {key.removeprefix("__input__") for key in archive.files if key.startswith("__input__")} != set(inputs):
        raise ValueError("Recurrent trace input set changed")
      initial = inputs
      inputs = {key: archive[f"__input__{key}"].copy() for key in initial}
      input_lineage = {}
      for key, value in inputs.items():
        if value.shape != initial[key].shape or value.dtype != initial[key].dtype:
          raise ValueError(f"Recurrent trace input layout changed: {key}")
        if key.startswith("state_"):
          output = "next_" + key
          if value.dtype == np.uint8:
            wanted = manifest["models"][name][label]["uint8_outputs"][frame - 1][output]
            input_lineage[key] = {"passed": output_record({key: value})[key] == wanted}
          else:
            input_lineage[key] = compare(value, golden[output][frame - 1])
        else:
          wanted = np.roll(initial[key], frame * 17, axis=-1) if key == "new_img" else initial[key]
          input_lineage[key] = compare(value, wanted)
  else:
    with np.load(references / "sequences/camera-inputs.npz") as images:
      inputs["new_img" if road else "input_img"][:] = images["rear" if road else "driver"][frame]
  if road and not args.recurrent:
    for key in inputs:
      if key.startswith("state_"):
        inputs[key].fill(0)
    inputs["desire"].fill(0)
    inputs["action_t"][:] = manifest["replay"]["action_t"]
    inputs["traffic_convention"][:] = [[0., 1.]]
  elif not road:
    inputs["calib"][:] = manifest["replay"]["device_from_calib_euler"]
  graph = infer_trace_types(onnx.load(source))
  types = {value.name: value.type.tensor_type.elem_type
           for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
  outputs = {value.name for value in graph.graph.output}
  nodes = []
  for index, node in enumerate(graph.graph.node):
    for key in node.output:
      if key and types.get(key):
        nodes.append({"index": index, "operator": node.op_type, "name": node.name, "output": key})
        if key not in outputs:
          graph.graph.output.append(onnx.helper.make_tensor_value_info(key, types[key], None))
          outputs.add(key)
  options = ort.SessionOptions()
  options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
  options.intra_op_num_threads, options.inter_op_num_threads = 2, 1
  session = ort.InferenceSession(graph.SerializeToString(), options, providers=["CPUExecutionProvider"])
  expected = dict(zip((value.name for value in session.get_outputs()), session.run(None, inputs), strict=True))
  with np.load(references / f"sequences/{name}-{label}.npz") as golden:
    hashes = manifest["models"][name][label]["uint8_outputs"][frame]
    host_final = compare_replay_frame({key: expected[key] for key in set(golden.files) | set(hashes)}, golden, hashes, frame)
  with np.load(args.archive) as actual:
    checks = [{**node, **compare(actual[node["output"]], expected[node["output"]], allow_masked_infinity=True)}
              for node in nodes if node["output"] in actual]
    first = next((node for node in checks if not node["passed"]), None)
    first_difference = next((node for node in checks if not node["exact"]), None)
    if first:
      np.savez_compressed(args.report.with_suffix(".first-divergence.npz"),
                          actual=actual[first["output"]], expected=expected[first["output"]])
    if first_difference:
      np.savez_compressed(args.report.with_suffix(".first-difference.npz"),
                          actual=actual[first_difference["output"]], expected=expected[first_difference["output"]])
    report = {"diagnostic_only": True, "vehicle_tx_enabled": False, "frame": frame, "seed": seed, "sequence": label,
              "input_lineage": input_lineage,
              "original_sha256": metadata["original_sha256"], "host_fusion_sha256": digest(source),
              "archive_sha256": digest(args.archive), "host_final_original_parity": host_final,
              "first_difference": first_difference, "first_divergence": first,
              "missing_outputs": [node["output"] for node in nodes if node["output"] not in actual], "nodes": checks}
  args.report.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({key: value for key, value in report.items() if key != "nodes"}, indent=2))


if __name__ == "__main__":
  main()
