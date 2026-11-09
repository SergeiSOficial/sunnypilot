#!/usr/bin/env python3
"""Run the Android arithmetic against unchanged host goldens on the host GPU.

This is a numerical regression check. Only device qualification can select a
live OpenCL backend; host timing and trace results cannot qualify the phone.
"""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tinygrad_repo"))
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
from fold3_model_trace import compare
from fold3_opencl import OpenCLModel, code_digest
from fold3_sequences import SEEDS, digest, initial_inputs, validate


def check(model, inputs, expected):
  actual = model(inputs)
  if set(actual) != set(expected):
    raise ValueError("Output names differ from the original reference")
  return {key: {**compare(value, expected[key], allow_masked_infinity=True),
                "exact": bool(np.array_equal(value, expected[key]))} for key, value in actual.items()}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--references", type=Path, default=ROOT / "android/app/build/model-reference")
  parser.add_argument("--trace-directory", type=Path)
  parser.add_argument("--skip-sequences", action="store_true", help="Diagnostic only; does not complete the host numerical gate")
  parser.add_argument("--model", action="append", choices=("dmonitoring_model", "driving_supercombo"))
  parser.add_argument("--device", default="METAL")
  parser.add_argument("--report", type=Path, required=True)
  args = parser.parse_args()
  from tinygrad.helpers import Context
  report = {"scope": "host GPU numerical regression only; not Android qualification", "device": args.device,
            "host": platform.platform(), "rtol": 1e-3, "atol": 1e-4, "vehicle_tx_enabled": False,
            "references_sha256": digest(args.references / "manifest.json"),
            "source_sha256": {key: code_digest(key) for key in ("fold3_opencl", "fold3_host_math")}, "models": {}}
  try:
    with Context(FLOAT16=0, IMAGE=0, BEAM=0):
      for name in args.model or ("dmonitoring_model", "driving_supercombo"):
        started = time.monotonic()
        graph = (args.trace_directory or args.references / "host-fusion") / f"{name}.onnx"
        model = OpenCLModel(graph, host_fusion=True, device=args.device)
        entry = {"graph_sha256": digest(graph), "seeds": {}}
        report["models"][name] = entry
        for seed in (42,) if args.trace_directory else SEEDS:
          suffix = "" if seed == 42 else f"-case-{seed}"
          with np.load((args.trace_directory or args.references) / f"{name}{suffix}-outputs.npz") as expected:
            entry["seeds"][str(seed)] = check(model, initial_inputs(args.references, name, seed), expected)
          print(name, "seed", seed, "failed outputs", [key for key, value in entry["seeds"][str(seed)].items()
                                                           if not value["passed"]], flush=True)
        if not args.trace_directory and not args.skip_sequences:
          entry["sequences"] = validate(model, args.references, name,
            progress=lambda: None)
          print(name, "sequences", {key: value["passed"] for key, value in entry["sequences"].items()}, flush=True)
        entry["seconds"] = time.monotonic() - started
        entry["status"] = "passed" if (all(value["passed"] for seed in entry["seeds"].values() for value in seed.values())
          and all(value["passed"] for value in entry.get("sequences", {}).values())) else "failed"
    report["status"] = "passed" if all(value["status"] == "passed" for value in report["models"].values()) else "failed"
    report["complete_host_gate"] = not args.trace_directory and not args.skip_sequences and len(report["models"]) == 2
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    raise
  finally:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
  return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
  raise SystemExit(main())
