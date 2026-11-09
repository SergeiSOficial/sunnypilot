#!/usr/bin/env python3
"""Collect operator diagnostics; completing a trace does not establish model parity."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile
import io
import time

from experiment import Device


def collect(device, destination, profile="trace", trace_directory=None, artifact_directory=None):
  if trace_directory is not None:
    if profile != "trace-qnn":
      raise ValueError("Uploaded traces require the diagnostic QNN profile")
    folder = Path(trace_directory)
    manifest = json.loads((folder / "manifest.json").read_text())
    names = {"manifest.json"}
    for name in manifest["models"]:
      if name not in ("dmonitoring_model", "driving_supercombo"):
        raise ValueError("Unknown trace model")
      names.update((f"{name}.onnx", f"{name}-outputs.npz"))
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as output:
      for name in sorted(names):
        output.add(folder / name, arcname=name)
    device.command("shell", "run-as", device.package, "rm", "-rf", "files/qnn-trace")
    device.command("shell", "run-as", device.package, "mkdir", "-p", "files/qnn-trace")
    subprocess.run([device.adb, "-s", device.serial, "exec-in", "run-as", device.package,
                    "tar", "-xf", "-", "-C", "files/qnn-trace"], input=archive.getvalue(), check=True, timeout=120)
  device.command("shell", "run-as", device.package, "rm", "-f", "files/model-report.json")
  device.command("shell", "am", "start", "-n", device.package + "/ai.sunnypilot.fold3.MainActivity",
                 "--ez", "run_models", "true", "--es", "model_profile", profile)
  deadline = time.monotonic() + 600
  required = {"ort", "qnn-gpu"} if profile == "trace-qnn" else {"ort", "CL"} if profile in ("trace", "trace-accum", "trace-fused") else {"ort"}
  observed_process = False
  while time.monotonic() < deadline:
    try:
      report = json.loads(device.command("exec-out", "run-as", device.package, "cat", "files/model-report.json"))
      Path(destination).write_text(json.dumps(report, indent=2) + "\n")
      if report.get("status") in ("blocked", "failed") and report.get("completed_utc"):
        raise RuntimeError(report.get("error", "Operator diagnostic failed"))
      if report.get("completed") and report.get("models") and all(
          set(model.get("engines", {})) == required for model in report["models"].values()):
        assert report["vehicle_tx_enabled"] is False
        if any("error" in trace for model in report["models"].values() for trace in model["engines"].values()):
          raise RuntimeError("Operator diagnostic contains an engine error; see retained report")
        if artifact_directory is not None:
          folder = Path(artifact_directory)
          folder.mkdir(parents=True, exist_ok=True)
          for name, model in report["models"].items():
            for engine, trace in model["engines"].items():
              paths = [trace.get("profile_path"), trace.get("provider_options", {}).get("profiling_file_path")]
              if trace.get("first_divergence"):
                paths.append(f"files/model-trace-{name}-{engine}.npz")
              for path in filter(None, paths):
                # Only read this application's trace artifacts, never arbitrary device paths.
                relative = "files/" + str(path).split("/files/", 1)[1] if "/files/" in str(path) else str(path)
                if not relative.startswith("files/") or ".." in Path(relative).parts:
                  raise ValueError("Trace artifact outside app files")
                data = subprocess.run([device.adb, "-s", device.serial, "exec-out", "run-as", device.package,
                                       "cat", relative], capture_output=True, check=True, timeout=60).stdout
                (folder / Path(relative).name).write_bytes(data)
        return report
    except (subprocess.CalledProcessError, json.JSONDecodeError):
      pass
    running = device.package + ":models" in device.command("shell", "ps", "-A", "-o", "NAME").split()
    observed_process |= running
    if observed_process and not running:
      raise RuntimeError("Model service exited before completing the operator trace")
    time.sleep(2)
  raise TimeoutError("Operator trace did not finish; partial diagnostic retained")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--package", default="com.samsung.sunnypilot.fold3")
  parser.add_argument("--report", required=True)
  parser.add_argument("--profile", default="trace", choices=("trace", "trace-cpu", "trace-accum", "trace-tail", "trace-fused", "trace-qnn"))
  parser.add_argument("--trace-directory", type=Path, help="Upload an isolated diagnostic export; original assets remain untouched")
  parser.add_argument("--artifact-directory", type=Path, help="Retain provider profiles and the first divergent tensor")
  args = parser.parse_args()
  result = collect(Device(args.serial, args.adb, args.package), args.report, args.profile, args.trace_directory, args.artifact_directory)
  print(json.dumps({name: {engine: {key: value for key, value in trace.items() if key in
                    ("error", "first_divergence", "first_nonfinite")} for engine, trace in model["engines"].items()}
                    for name, model in result["models"].items()}, indent=2))
