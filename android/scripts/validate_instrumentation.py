#!/usr/bin/env python3
"""Require a completed JUnit suite; am instrument can exit zero after a crash."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from experiment import PACKAGE, CAMERA_COMPAT_PACKAGE


def completed_tests(output):
  result = re.search(r"^OK \((\d+) tests?\)\s*$", output, re.MULTILINE)
  if not result or any(text in output for text in ("FAILURES!!!", "INSTRUMENTATION_FAILED", "INSTRUMENTATION_ABORTED", "shortMsg=")):
    return 0
  return int(result[1])


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--package", choices=(PACKAGE, CAMERA_COMPAT_PACKAGE), default=PACKAGE)
  parser.add_argument("--expected-tests", type=int, default=17)
  parser.add_argument("--report", required=True)
  parser.add_argument("--output", help="Save the unmodified instrumentation output separately from the parsed result")
  args = parser.parse_args()
  command = [args.adb, "-s", args.serial, "shell", "am", "instrument", "-w",
             args.package + ".test/androidx.test.runner.AndroidJUnitRunner"]
  result = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
  output = result.stdout + result.stderr
  if args.output:
    Path(args.output).write_text(output)
  tests = completed_tests(output)
  report = {"serial": args.serial, "package": args.package, "completed_tests": tests, "expected_tests": args.expected_tests,
            "returncode": result.returncode, "status": "passed" if result.returncode == 0 and tests == args.expected_tests else "failed"}
  Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
  print(output, end="")
  raise SystemExit(0 if report["status"] == "passed" else 1)
