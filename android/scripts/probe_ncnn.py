#!/usr/bin/env python3
"""Isolated Fold 3 Vulkan correctness diagnostic; never backend qualification."""
import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from experiment import Device, device_lease
from fold3_model_trace import compare
from fold3_sequences import initial_inputs
from prepare_accel_models import mark_ncnn_half_heads, repair_ncnn_casts, restore_ncnn_half_outputs, mark_ncnn_reference_layernorm, mark_ncnn_float_gemm
from ncnn_replay_inputs import replay_case


def sha(path):
  with Path(path).open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def unfuse_activations(graph, operators):
  """Expose a fused activation for diagnosis without changing graph outputs."""
  lines = graph.read_text().splitlines()
  requested = set(operators)
  if not requested:
    return []
  if len(requested) != len(operators):
    raise ValueError("Duplicate unfused operator")
  records, body = [], []
  for line in lines[2:]:
    words = line.split()
    if words[1] in requested:
      if words[0] not in ("Convolution", "ConvolutionDepthWise", "InnerProduct") or words[2:4] != ["1", "1"]:
        raise ValueError("Unfusing requires a single-input weighted operation")
      attributes = dict(word.split("=", 1) for word in words[6:])
      activation = {"1": "ReLU", "4": "Sigmoid"}.get(attributes.get("9"))
      if activation is None or "-23310" in attributes:
        raise ValueError("Unsupported fused activation")
      original_output = words[5]
      intermediate = "fold3_preactivation_" + words[1]
      generated_operator = "fold3_activation_" + words[1]
      if any(intermediate in row.split()[4:] or row.split()[1] == generated_operator for row in lines[2:]):
        raise ValueError("Unfused diagnostic name collision")
      words[5] = intermediate
      words[6:] = [word for word in words[6:] if not word.startswith("9=")]
      body.extend([" ".join(words), f"{activation} fold3_activation_{words[1]} 1 1 {intermediate} {original_output}"])
      records.append({"operator": words[1], "activation": activation,
                      "activation_operator": "fold3_activation_" + words[1]})
    else:
      body.append(line)
  if {record["operator"] for record in records} != requested:
    raise ValueError("Unfused operator not found")
  layers, blobs = map(int, lines[1].split())
  graph.write_text("\n".join([lines[0], f"{layers + len(records)} {blobs + len(records)}", *body]) + "\n")
  return records


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--report", required=True, type=Path)
  parser.add_argument("--model", choices=("dmonitoring_model", "driving_supercombo"), default="dmonitoring_model")
  parser.add_argument("--seed", type=int, choices=(42, 7, 123), default=42)
  parser.add_argument("--replay-frame", type=int, choices=range(60))
  parser.add_argument("--trace-count", type=int, default=12, choices=range(65))
  parser.add_argument("--trace-skip", type=int, default=0, choices=range(4097))
  parser.add_argument("--trace-stride", type=int, default=1, choices=range(1, 65))
  parser.add_argument("--trace-weighted", action="store_true", help="Trace only Conv/depthwise/linear outputs, retaining blob indices")
  parser.add_argument("--ordered-half-node", action="append", metavar="ONNX_NODE",
                      help="Use the ordered Vulkan half GEMV for a weight-verified original node")
  parser.add_argument("--ordered-half-dot2", action="store_true",
                      help="Diagnostic original NoTrans half GEMM accumulation and output rounding")
  parser.add_argument("--round-half-outputs", action="store_true",
                      help="Restore weight-verified original Gemm and LayerNorm half output boundaries")
  parser.add_argument("--reference-layernorm-node", action="append", default=[], metavar="ONNX_NODE",
                      help="Diagnostic Vulkan reference arithmetic for a provenance-verified FP16 LayerNorm")
  parser.add_argument("--reference-float-gemm-node", action="append", default=[], metavar="ONNX_NODE")
  parser.add_argument("--unfuse-activation", action="append", default=[], metavar="NCNN_OPERATOR",
                      help="Expose a fused ReLU/Sigmoid preactivation in a separately hashed diagnostic graph")
  args = parser.parse_args()
  if args.replay_frame is not None and args.model != "dmonitoring_model":
    parser.error("Road replay traces require actual recurrent warmup")
  if args.report.exists():
    raise FileExistsError(args.report)
  folder = args.report.with_suffix("")
  folder.mkdir(parents=True, exist_ok=False)
  references = ROOT / "android/app/build/model-reference"
  native = ROOT / "android/build/native/accelerators/ncnn-patched"
  stamp = json.loads((native / "build.json").read_text())
  if (stamp["patch_sha256"] != sha(ROOT / "android/patches/ncnn-explicit-casts.patch") or
      stamp["recipe_sha256"] != sha(Path(__file__).with_name("build_ncnn.py"))):
    raise ValueError("Stale patched ncnn build")
  metadata = json.loads((references / "accelerators/manifest.json").read_text())["models"][args.model]
  adapter = metadata["ncnn"]
  source = references / f"accelerators/ncnn/{args.model}.ncnn.param"
  if sha(source) != adapter["sha256"] or sha(source.with_suffix(".bin")) != adapter["weights_sha256"]:
    raise ValueError("Staged Vulkan graph or weights do not match the manifest")
  graph = folder / "model.param"
  graph.write_bytes(source.read_bytes())
  repairs = adapter.get("cast_repairs")
  if repairs is None:
    repairs = repair_ncnn_casts(graph, source.with_name(f"{args.model}.pnnx.param"))
  half_gemv = None
  if args.ordered_half_node or args.round_half_outputs or args.ordered_half_dot2 or args.reference_layernorm_node or args.reference_float_gemm_node:
    graph.with_suffix(".bin").symlink_to(source.with_suffix(".bin"))
  if args.ordered_half_node:
    original = ROOT / f"openpilot/selfdrive/modeld/models/{args.model}.onnx"
    if sha(original) != metadata["source_sha256"]:
      raise ValueError("Original graph does not match the staged reference")
    half_gemv = mark_ncnn_half_heads(graph, original, original_nodes=args.ordered_half_node)
  half_outputs = restore_ncnn_half_outputs(graph, references / f"accelerators/{args.model}.onnx") if args.round_half_outputs else None
  half_dot2 = restore_ncnn_half_outputs(graph, references / f"accelerators/{args.model}.onnx", mode="dot2") if args.ordered_half_dot2 else None
  reference_gemm = (mark_ncnn_float_gemm(graph, references / f"accelerators/{args.model}.onnx",
                    args.reference_float_gemm_node) if args.reference_float_gemm_node else None)
  if reference_gemm and stamp.get("reference_float_gemm") != 1:
    raise ValueError("Native build lacks reference FP32 Gemm capability")
  reference_norm = (mark_ncnn_reference_layernorm(graph, references / f"accelerators/{args.model}.onnx",
                    args.reference_layernorm_node) if args.reference_layernorm_node else None)
  if reference_norm and stamp.get("reference_layernorm") != 1:
    raise ValueError("Patched ncnn lacks reference LayerNorm capability")
  unfused = unfuse_activations(graph, args.unfuse_activation)
  sdk = Path.home() / "Library/Android/sdk"
  toolchain = sdk / f"ndk/{stamp['ndk']}/toolchains/llvm/prebuilt/darwin-x86_64"
  cpp = Path(__file__).with_name("ncnn_probe.cpp")
  binary = folder / "f3ncnnprobe"
  install = native / "install"
  subprocess.run([str(toolchain / "bin/aarch64-linux-android35-clang++"), "-std=c++17", "-O2",
                  "-I" + str(install / "include/ncnn"), str(cpp), "-L" + str(install / "lib"),
                  "-lncnn", "-o", str(binary)], check=True)
  if args.replay_frame is None:
    inputs = initial_inputs(references, args.model, args.seed)
    suffix = "" if args.seed == 42 else f"-case-{args.seed}"
    with np.load(references / f"{args.model}{suffix}-outputs.npz") as expected:
      golden = {key: expected[key].copy() for key in metadata["outputs"]}
  else:
    inputs, expected = replay_case(references, args.replay_frame)
    golden = {"outputs": expected}
  shapes = []
  input_files = []
  for index, (key, contract) in enumerate(metadata["inputs"].items()):
    value = inputs[key].reshape(adapter.get("engine_inputs", {}).get(key, contract)["shape"])
    axis = adapter["input_batch_axes"][key]
    if axis is not None:
      value = np.moveaxis(value, axis, 0)
    batch, dims = (value.shape[0], value.shape[1:]) if axis is not None else (1, value.shape)
    name = f"in{index}"
    shapes.append(" ".join(map(str, (name, len(dims), batch, *reversed(dims)))))
    np.ascontiguousarray(value, dtype=np.float32).tofile(folder / f"{name}.bin")
    input_files.append((folder / f"{name}.bin", f"{name}.bin"))
  (folder / "shapes.txt").write_text("\n".join([*shapes, str(len(golden))]) + "\n")
  files = [(binary, "f3ncnnprobe"), (graph, "model.param"),
           (source.with_suffix(".bin"), "model.bin"), (install / "lib/libncnn.so", "libncnn.so"),
           (toolchain / "sysroot/usr/lib/aarch64-linux-android/libc++_shared.so", "libc++_shared.so"),
           (folder / "shapes.txt", "shapes.txt"), *input_files]
  device = Device(args.serial, args.adb, "com.samsung.sunnypilot.fold3")
  directory = f"/data/local/tmp/fold3-ncnn-{time.time_ns()}"
  report = {"status": "running", "diagnostic_only": True, "qualified": False, "vehicle_tx_enabled": False,
            "model": args.model, "seed": args.seed, "trace_count": args.trace_count, "trace_skip": args.trace_skip, "trace_mode": "weighted" if args.trace_weighted else "all", "cast_repairs": repairs, "build": stamp,
            "source_graph_sha256": sha(source), "ordered_half_gemv": half_gemv,
            "unfused_activations": unfused, "half_output_repairs": half_outputs, "half_dot2_repairs": half_dot2,
            "reference_layernorm": reference_norm, "reference_float_gemm": reference_gemm,
            "sha256": {name: sha(path) for path, name in files},
            "probe_sha256": sha(cpp), "runner_sha256": sha(Path(__file__))}
  report["replay_frame"] = args.replay_frame
  report["trace_stride"] = args.trace_stride
  with device_lease(args.serial):
    created, process = False, None
    try:
      report["forwards_before"] = device.forwards()
      report["after_stop"] = device.stop()
      report["baseline"] = device.cool()
      report["apk_sha256"] = device.artifact()
      report["fingerprint"] = device.command("shell", "getprop", "ro.build.fingerprint").strip()
      device.command("shell", "mkdir", directory)
      created = True
      for path, name in files:
        device.command("push", str(path), directory + "/" + name)
      device.command("shell", "chmod", "700", directory + "/f3ncnnprobe")
      process = subprocess.Popen([device.adb, "-s", device.serial, "shell", "env", "LD_LIBRARY_PATH=" + directory,
                                  "timeout", "180", directory + "/f3ncnnprobe", directory, str(args.trace_count), str(args.trace_skip), str(args.trace_stride), "weighted" if args.trace_weighted else "all"],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      while True:
        device.assert_thermal_safe()
        try:
          stdout, stderr = process.communicate(timeout=1)
          break
        except subprocess.TimeoutExpired:
          pass
      (folder / "stdout.log").write_text(stdout)
      (folder / "stderr.log").write_text(stderr)
      if process.returncode:
        raise RuntimeError(f"Vulkan probe failed: {process.returncode}: {stderr[-1000:]}")
      capabilities = [int(line.split()[1]) for line in stdout.splitlines() if line.startswith("FOLD3_REFERENCE_LAYERNORM ")]
      report["native_reference_layernorm"] = capabilities[0] if len(capabilities) == 1 else None
      if reference_norm and report["native_reference_layernorm"] != 1:
        raise ValueError("Loaded native library lacks reference LayerNorm capability")
      capabilities = [int(line.split()[1]) for line in stdout.splitlines() if line.startswith("FOLD3_REFERENCE_FLOAT_GEMM ")]
      report["native_reference_float_gemm"] = capabilities[0] if len(capabilities) == 1 else None
      if reference_gemm and report["native_reference_float_gemm"] != 1:
        raise ValueError("Loaded native library lacks reference FP32 Gemm capability")
      tensors = []
      for line in stdout.splitlines():
        if line.startswith("FOLD3_TENSOR "):
          _, index, blob, operation, kind, w, h, depth, channels, batch = line.split()
          filename = f"trace-{int(index)}.bin"
          device.command("pull", directory + "/" + filename, str(folder / filename))
          tensors.append({"blob": blob, "operator": operation, "type": kind, "backend": "Vulkan",
                          "dtype": "float32", "layout": "NCDHW", "shape": list(map(int, (batch, channels, depth, h, w))),
                          "file": filename, "sha256": sha(folder / filename)})
      report["tensors"] = tensors
      checks = {}
      for index, (key, contract) in enumerate(metadata["outputs"].items()):
        filename = f"out{index}.bin"
        device.command("pull", directory + "/" + filename, str(folder / filename))
        shape = list(adapter.get("engine_outputs", {}).get(key, contract)["shape"])
        axis = adapter["output_batch_axes"][key]
        if axis is not None:
          shape.insert(0, shape.pop(axis))
        actual = np.fromfile(folder / filename, np.float32).reshape(shape)
        if axis is not None:
          actual = np.moveaxis(actual, 0, axis)
        checks[key] = compare(actual.reshape(contract["shape"]), golden[key])
      report["output_checks"] = checks
      if len(checks) == 1:
        report["parity"] = checks["outputs"]
      report["status"] = "passed" if all(check["passed"] for check in checks.values()) else "failed"
      report["rtol"], report["atol"] = 1e-3, 1e-4
    except Exception as error:
      report.update(status="blocked", error=f"{type(error).__name__}: {error}")
    finally:
      if created:
        try:
          pids = device.command("shell", "pidof", "f3ncnnprobe").split()
        except subprocess.CalledProcessError:
          pids = []
        for pid in pids:
          command = device.command("exec-out", "cat", f"/proc/{int(pid)}/cmdline")
          if command.split("\0")[0] == directory + "/f3ncnnprobe":
            device.command("shell", "kill", "-9", str(int(pid)))
        if process is not None and process.poll() is None:
          process.kill()
          process.communicate(timeout=10)
        # Only this invocation's unique directory is removed.
        device.command("shell", "rm", "-rf", directory)
      try:
        report["after_cleanup"] = device.stop()
        report["forwards_removed"] = device.clean_forwards(report.get("forwards_before", []))
        report["apk_sha256_after"] = device.artifact()
        report["artifact_stable"] = report.get("apk_sha256") == report["apk_sha256_after"]
        report["final_baseline"] = device.cool()
      except Exception as error:
        report.update(status="blocked", cleanup_error=f"{type(error).__name__}: {error}")
      finally:
        report["completed_utc"] = datetime.now(UTC).isoformat()
        args.report.write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({key: report.get(key) for key in ("status", "parity", "error")}))
  return 0 if report["status"] == "passed" and report.get("artifact_stable") else 1


if __name__ == "__main__":
  raise SystemExit(main())
