#!/usr/bin/env python3
"""Run intact accuracy sequences as separate cooled, artifact-bound experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
REPORT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "android/app/src/main/python"))
sys.path.insert(0, str(ROOT / "android/scripts"))
from fold3_accel import ACCURACY_PARTS, accuracy_part_passed, combine_accuracy_parts
from experiment import Device


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("candidate")
  parser.add_argument("--serial", default="RFCR81Y1TBJ")
  parser.add_argument("--adb", default="/Users/sergios/Library/Android/sdk/platform-tools/adb")
  parser.add_argument("--models", nargs="+", choices=("road", "driver"), default=("road", "driver"))
  args = parser.parse_args()
  artifact = json.loads((ROOT / "android/build/candidates" / args.candidate / "artifacts.json").read_text())
  apk = artifact["files"]["app-debug.apk"]["sha256"]
  if Device(args.serial, args.adb, "com.samsung.sunnypilot.fold3").artifact() != apk:
    raise RuntimeError("Installed APK differs from the requested immutable candidate")
  suffix = args.candidate.removeprefix("candidate")
  failed = False
  for model in args.models:
    parts = {}
    name = "driving_supercombo" if model == "road" else "dmonitoring_model"
    for part in ACCURACY_PARTS:
      stem = f"fold{suffix}-mnn-{model}-accuracy-{part}"
      destination, isolation_path = REPORT / (stem + ".json"), REPORT / (stem + "-isolation.json")
      if destination.exists() or isolation_path.exists():
        raise FileExistsError("Preserve previous evidence; use a fresh candidate or reviewed retry inventory")
      command = [sys.executable, "android/scripts/experiment.py", "--serial", args.serial, "--adb", args.adb,
        "--package", "com.samsung.sunnypilot.fold3", "--report", str(isolation_path), "--", sys.executable,
        "android/scripts/validate_models.py", "--serial", args.serial, "--adb", args.adb,
        "--package", "com.samsung.sunnypilot.fold3", "--profile", f"mnn-buffer-accuracy-{model}-{part}",
        "--timeout", "7200", "--report", str(destination)]
      print(f"Starting {model} {part}", flush=True)
      with (REPORT / (stem + "-command.log")).open("w") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
      isolation = json.loads(isolation_path.read_text())
      if (isolation.get("isolation_status") != "passed" or isolation.get("artifact_stable") is not True or
          isolation.get("apk_sha256") != apk):
        raise RuntimeError(f"Isolation or artifact identity failed: {isolation_path.name}")
      report = json.loads(destination.read_text())
      entry = report.get("models", {}).get(name, {})
      if (result.returncode or report.get("validation_status") != "passed" or
          entry.get("identity", {}).get("sha256", {}).get("apk") != apk or
          not accuracy_part_passed(entry, entry.get("identity"), part)):
        print(f"Failed {model} {part}; remaining parts of this model not advanced", flush=True)
        failed = True
        break
      parts[part] = entry
      print(f"Passed {model} {part}", flush=True)
    if len(parts) == len(ACCURACY_PARTS):
      combined = combine_accuracy_parts(parts, parts["initial"]["identity"])
      if combined is None:
        raise RuntimeError("Host could not verify the complete original-reference aggregate")
      if report.get("accuracy_aggregate", {}).get(name) != combined:
        raise RuntimeError("Device and host accuracy aggregates differ")
      final = {"candidate": args.candidate, "vehicle_tx_enabled": False, "validation_status": "passed",
        "stage": "accuracy", "models": {name: combined}, "part_evidence": []}
      for part in ACCURACY_PARTS:
        source = REPORT / f"fold{suffix}-mnn-{model}-accuracy-{part}.json"
        final["part_evidence"].append({"evidence": source.name,
          "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "isolation": source.stem + "-isolation.json"})
      (REPORT / f"fold{suffix}-mnn-{model}-accuracy-complete.json").write_text(json.dumps(final, indent=2) + "\n")
      print(f"Complete original-reference accuracy passed: {model}", flush=True)
  return 1 if failed else 0


if __name__ == "__main__":
  raise SystemExit(main())
