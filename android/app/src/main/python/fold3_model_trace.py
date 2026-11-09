"""Operator diagnostics; results cannot select a production inference backend."""
import hashlib
import json
from pathlib import Path


def compare(actual, expected, allow_masked_infinity=False):
  import numpy as np
  finite = bool(np.isfinite(actual).all())
  if actual.shape != expected.shape:
    return {"passed": False, "finite": finite, "shape": list(actual.shape), "expected_shape": list(expected.shape),
            "dtype": str(actual.dtype), "max_abs_error": None, "mismatched_elements": None}
  finite_pair = np.isfinite(actual) & np.isfinite(expected)
  unexpected_nonfinite = ~np.isfinite(actual)
  if allow_masked_infinity:
    unexpected_nonfinite &= ~(np.isneginf(actual) & np.isneginf(expected))
  difference = np.abs(actual[finite_pair].astype(np.float64) - expected[finite_pair].astype(np.float64))
  # Attention masks deliberately contain -inf. Match their positions/signs;
  # a new infinity or any NaN still fails. Final model qualification is finite-only.
  return {"passed": bool(not unexpected_nonfinite.any() and np.allclose(actual, expected, rtol=1e-3, atol=1e-4)),
          "finite": finite, "shape": list(actual.shape), "dtype": str(actual.dtype),
          "expected_nonfinite_elements": int(np.count_nonzero(~np.isfinite(expected))),
          "unexpected_nonfinite_elements": int(np.count_nonzero(unexpected_nonfinite)),
          "exact": bool(np.array_equal(actual, expected)),
          "max_abs_error": float(difference.max(initial=0)),
          "mismatched_elements": int(np.count_nonzero(~np.isclose(actual, expected, rtol=1e-3, atol=1e-4)))}


def compare_replay_frame(actual, expected, hashes, frame):
  """Check numerical outputs and the separately hashed uint8 image history."""
  import numpy as np
  from fold3_sequences import output_record
  if set(actual) != set(expected.files) | set(hashes):
    raise ValueError("Replay trace output set changed")
  checks = {key: compare(value, expected[key][frame]) for key, value in actual.items() if value.dtype != np.uint8}
  exact = output_record(actual)
  checks.update({key: {"passed": exact.get(key) == value, "exact_hash": exact.get(key)} for key, value in hashes.items()})
  return checks


def half_conversion_diagnostic(device="CL"):
  """Exercise every finite half value and neighbors of rounding boundaries."""
  import numpy as np
  from tinygrad import Tensor, dtypes
  halves = np.arange(65536, dtype=np.uint16).view(np.float16)
  finite = halves[np.isfinite(halves)].astype(np.float32)
  edges = np.array([2**-25, 3 * 2**-25, 2**-14 - 2**-25, 65520.], dtype=np.float32)
  edges = np.concatenate((edges, -edges))
  values = np.concatenate((finite, edges, np.nextafter(edges, -np.inf), np.nextafter(edges, np.inf)))
  with np.errstate(over="ignore"):
    expected = values.astype(np.float16)
  actual = Tensor(values, device=device).cast(dtypes.float16).numpy()
  mismatch = np.flatnonzero(actual.view(np.uint16) != expected.view(np.uint16))
  return {"passed": not len(mismatch), "values": len(values), "mismatched_elements": len(mismatch),
          "subnormal_values": int(np.count_nonzero((expected != 0) & (np.abs(expected) < np.finfo(np.float16).tiny))),
          "first_mismatches": [{"input": float(values[i]), "actual_bits": int(actual.view(np.uint16)[i]),
                                "expected_bits": int(expected.view(np.uint16)[i])} for i in mismatch[:8]]}


def gemm_diagnostic(device="CL", on_result=None):
  """Compare device half rounding with an exact float64 lane reference."""
  import numpy as np
  from types import SimpleNamespace
  from tinygrad import Tensor
  from fold3_opencl import fused_ops
  runner = SimpleNamespace(onnx_ops={})
  fused_ops(runner)
  checks = []
  for seed in (42, 7, 123):
    random = np.random.default_rng(seed)
    for length in (8, 19, 64, 512, 1024, 513):
      a = random.normal(size=(1, length)).astype(np.float16)
      b = random.normal(size=(32, length)).astype(np.float16)
      if length == 513:
        b *= np.float16(0.015625)
      c = random.normal(size=(32,)).astype(np.float16)
      end = length // 8 * 8
      lanes = np.zeros((32, 8), dtype=np.float16)
      for index in range(0, end, 8):
        lanes = (a[:, index:index + 8].astype(np.float64) * b[:, index:index + 8].astype(np.float64)
                 + lanes.astype(np.float64)).astype(np.float16)
      for width in (4, 2, 1):
        lanes = (lanes[:, ::2] + lanes[:, 1::2]).reshape(32, width)
      expected = lanes.reshape(1, 32)
      for index in range(end, length):
        expected = (expected + (a[0, index] * b[:, index]).astype(np.float16)).astype(np.float16)
      expected = (expected + c).astype(np.float16)
      actual = runner.onnx_ops["Gemm"](*(Tensor(value, device=device) for value in (a, b, c)), transB=1).numpy()
      checks.append({"seed": seed, "length": length, **compare(actual, expected),
                     "exact": bool(np.array_equal(actual, expected))})
      if on_result:
        on_result(checks)
  return checks


def trace_camera(data_dir, road=False, recurrent=False):
  """Observe a replay or recurrent input without changing the model graph."""
  import numpy as np
  from fold3_models import _thermal_check
  from fold3_opencl import OpenCLModel, identity
  from fold3_sequences import digest, feedback, initial_inputs, sequence_input
  import time
  started = time.time()  # noqa: TID251 - UTC audit metadata, not elapsed-time measurement.
  references = Path(__file__).parent / "fold3_reference"
  if recurrent and not road:
    raise ValueError("Recurrent trace requires the road model")
  name, frame = ("driving_supercombo", 10 if recurrent else 0) if road else ("dmonitoring_model", 41)
  seed = 7 if recurrent else 42
  label = f"seed{seed}" if recurrent else "camera-replay"
  folder = references / "sequences"
  manifest = json.loads((folder / "manifest.json").read_text())
  for asset, expected in manifest["assets"].items():
    if digest(folder / asset) != expected:
      raise ValueError(f"Sequence asset hash mismatch: {asset}")
  path = references / "host-fusion" / f"{name}.onnx"
  metadata = json.loads((path.parent / "manifest.json").read_text())["models"][name]
  original = Path(__file__).parent / "openpilot/selfdrive/modeld/models" / f"{name}.onnx"
  if digest(path) != metadata["sha256"] or digest(original) != metadata["original_sha256"]:
    raise ValueError("Replay trace model hash mismatch")
  initial = initial_inputs(references, name, seed)
  inputs = {key: value.copy() for key, value in initial.items()}
  if not recurrent:
    with np.load(folder / "camera-inputs.npz") as images:
      inputs["new_img" if road else "input_img"][:] = images["rear" if road else "driver"][frame]
  if road and not recurrent:
    for key in inputs:
      if key.startswith("state_"):
        inputs[key].fill(0)
    inputs["desire"].fill(0)
    inputs["action_t"][:] = manifest["replay"]["action_t"]
    inputs["traffic_convention"][:] = [[0., 1.]]
  elif not road:
    inputs["calib"][:] = manifest["replay"]["device_from_calib_euler"]
  _thermal_check()
  model = OpenCLModel(path, host_fusion=True)
  observed = None
  if recurrent:
    for index in range(frame + 1):
      _thermal_check()
      sequence_input(inputs, initial, index)
      observed = model(inputs)
      if index < frame:
        feedback(inputs, observed)
    # TinyJit retains graph_values from its capture invocation, rather than
    # the most recent replay. A fresh runner exposes the actual target input.
    model = OpenCLModel(path, host_fusion=True)
  actual = model(inputs)
  with np.load(folder / f"{name}-{label}.npz") as expected:
    hashes = manifest["models"][name][label]["uint8_outputs"][frame]
    final = compare_replay_frame(actual, expected, hashes, frame)
  destination = Path(data_dir)
  archive = destination / f"model-camera-trace-{name}.npz"
  profile = "trace-road-seed7" if recurrent else "trace-road-camera" if road else "trace-camera"
  report = {"diagnostic_only": True, "started_utc": started, "profile": profile,
            "scope": "Unmodified graph, retained operator tensors; tensor reads can change scheduling",
            "vehicle_tx_enabled": False, "frame": frame, "seed": seed, "sequence": label, "models": {name: {
              "identity": identity(name, path, references), "final": final,
              "performance_status": "blocked", "status": "blocked", "archive": archive.name}}}
  if observed is not None:
    report["models"][name]["jit_reproduction"] = {key: compare(actual[key], value) for key, value in observed.items()}
  (destination / "model-report.json").write_text(json.dumps(report, indent=2))
  tensors = {f"__input__{key}": value for key, value in inputs.items()} if recurrent else {}
  if observed is not None:
    tensors.update({f"__observed__{key}": value for key, value in observed.items()})
  nodes = []
  for index, node in enumerate(model.runner.graph_nodes):
    _thermal_check()
    for key in node.outputs:
      value = model.runner.graph_values.get(key)
      if key and hasattr(value, "numpy"):
        tensors[key] = value.numpy()
        nodes.append({"index": index, "operator": node.op, "output": key})
  np.savez_compressed(archive, **tensors)
  report["models"][name].update(nodes=nodes, archive_sha256=digest(archive), archive_bytes=archive.stat().st_size)
  report["completed"] = True
  report["completed_utc"] = time.time()  # noqa: TID251 - UTC audit metadata, not elapsed-time measurement.
  (destination / "model-report.json").write_text(json.dumps(report, indent=2))
  return json.dumps(report)


def trace(data_dir, opencl=True, accumulation=False, directory="trace", qnn=False, library_dir=None):
  import numpy as np
  from fold3_ort import OrtModel
  from java import jclass
  hardware = jclass("ai.sunnypilot.fold3.AndroidHardware")
  references = Path(__file__).parent / "fold3_reference"
  uploaded = Path(data_dir) / "qnn-trace"
  directory = uploaded if qnn and (uploaded / "manifest.json").is_file() else references / directory
  manifest = json.loads((directory / "manifest.json").read_text())
  originals = json.loads((references / "manifest.json").read_text())["models"]
  report = {"scope": manifest["scope"], "diagnostic_only": True, "vehicle_tx_enabled": False, "models": {}}
  destination = Path(data_dir) / "model-report.json"
  if opencl and manifest.get("optimized"):
    report["opencl_capabilities"] = json.loads(str(jclass("ai.sunnypilot.fold3.NativeBridge").openClProbe()))
    destination.write_text(json.dumps(report, indent=2))
    try:
      report["half_conversion_check"] = half_conversion_diagnostic()
      destination.write_text(json.dumps(report, indent=2))
      def save_checks(checks):
        report["gemm_checks"] = checks
        destination.write_text(json.dumps(report, indent=2))
      report["gemm_checks"] = gemm_diagnostic(on_result=save_checks)
    except Exception as exc:
      report["gemm_error"] = str(exc)
    destination.write_text(json.dumps(report, indent=2))
  for name, metadata in manifest["models"].items():
    path = directory / f"{name}.onnx"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
      raise ValueError("Operator trace hash mismatch")
    if metadata["original_sha256"] != originals[name]["sha256"]:
      raise ValueError("Operator trace original model provenance mismatch")
    with np.load(references / f"{name}-inputs.npz") as source:
      inputs = dict(source)
    entry = report["models"][name] = {"sha256": metadata["sha256"], "original_sha256": metadata["original_sha256"],
      "instrumented_matches_original": metadata["instrumented_matches_original"], "engines": {}}
    for engine in ("ort", "qnn-gpu") if qnn else ("ort", "CL") if opencl else ("ort",):
      if hardware.thermalStatus() >= 2:
        raise RuntimeError("Operator diagnostics stopped at MODERATE thermal status")
      try:
        details = {}
        if engine in ("ort", "qnn-gpu"):
          provider_options = ({"profiling_level": "detailed", "profiling_file_path":
                               str(Path(data_dir) / f"trace-{name}-qnn.csv")} if engine == "qnn-gpu" else {})
          model = OrtModel(path, "cpu" if engine == "ort" else engine, threads=2,
                           strict=manifest.get("optimized", False), library_dir=library_dir, assignments=True,
                           provider_options=provider_options,
                           profile_path=Path(data_dir) / f"trace-{name}-{engine}")
          details["assignment"] = model.describe()
          actual = model(inputs)
          details["profile_path"] = model.end_profiling()
          details["provider_options"] = provider_options
        else:
          from tinygrad import Tensor
          from tinygrad.nn.onnx import OnnxRunner
          from tinygrad.helpers import Context
          with Context(FLOAT16=0, IMAGE=0, BEAM=0):
            model = OnnxRunner(Tensor(path.read_bytes(), device="PYTHON")).to("CL")
            if manifest.get("optimized"):
              from fold3_opencl import fused_ops
              fused_ops(model)
            if accumulation:
              from fold3_precision import accumulation_diagnostic
              accumulation_diagnostic(model)
            tensors = model({key: Tensor(value, device="CL") for key, value in inputs.items()})
            Tensor.realize(*tensors.values())
            actual = {key: value.numpy() for key, value in tensors.items()}
        with np.load(directory / f"{name}-outputs.npz") as expected:
          nodes = [{**node, **compare(actual[node["output"]], expected[node["output"]], allow_masked_infinity=True)}
                   for node in metadata["nodes"]]
          final = {key: compare(actual[key], expected[key]) for key in actual if key not in {node["output"] for node in nodes}}
          divergent = next((node for node in nodes if not node["passed"]), None)
          different = next((node for node in nodes if not node["exact"]), None)
          if divergent:
            np.savez_compressed(Path(data_dir) / f"model-trace-{name}-{engine}.npz",
                                actual=actual[divergent["output"]], expected=expected[divergent["output"]])
          if different:
            np.savez_compressed(Path(data_dir) / f"model-trace-{name}-{engine}-first-difference.npz",
                                actual=actual[different["output"]], expected=expected[different["output"]])
        entry["engines"][engine] = {**details, "nodes": nodes, "final": final,
          "first_difference": different,
          "first_divergence": next((node for node in nodes if not node["passed"]), None),
          "first_nonfinite": next((node for node in nodes if node.get("unexpected_nonfinite_elements")), None)}
      except Exception as exc:
        import traceback
        entry["engines"][engine] = {"error": str(exc), "traceback": traceback.format_exc()}
      destination.write_text(json.dumps(report, indent=2))
  report["completed"] = True
  destination.write_text(json.dumps(report, indent=2))
  return json.dumps(report)
