#!/usr/bin/env python3
"""Sample a diagnostic run without interpreting camera timing as model/control timing."""
import argparse
import json
import pathlib
import re
import subprocess
import time


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--serial", required=True)
  parser.add_argument("--process", default="ai.sunnypilot.fold3")
  parser.add_argument("--progress-file", choices=("progress.txt", "model-progress.txt"), default="progress.txt")
  parser.add_argument("--seconds", type=int, default=1900)
  parser.add_argument("--output", type=pathlib.Path, required=True)
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)

  def shell(*command):
    try:
      return subprocess.run([args.adb, "-s", args.serial, "shell", *command], capture_output=True, text=True, timeout=12).stdout
    except subprocess.TimeoutExpired:
      return "adb timeout"

  start = time.monotonic()
  with args.output.open("w") as output:
    while time.monotonic() - start < args.seconds:
      memory = shell("dumpsys", "meminfo", args.process)
      thermal = shell("dumpsys", "thermalservice")
      battery = shell("dumpsys", "battery")
      sample = {"elapsed_s": time.monotonic() - start, "utc": time.time(),  # noqa: TID251 -- correlate Android and host reports
                "process": args.process, "progress_file": args.progress_file,
                "progress": shell("run-as", "ai.sunnypilot.fold3", "cat", "files/" + args.progress_file),
                "memory_kib": {key: int(value) for key, value in re.findall(r"(TOTAL PSS|TOTAL RSS|TOTAL SWAP PSS):\s*(\d+)", memory)},
                "thermal": [line.strip() for line in thermal.splitlines() if "Thermal Status:" in line or "mType=3," in line],
                "battery": [line.strip() for line in battery.splitlines() if re.match(r"\s*(level|temperature|voltage):", line)]}
      output.write(json.dumps(sample) + "\n")
      output.flush()
      time.sleep(min(15, max(0, args.seconds - (time.monotonic() - start))))


if __name__ == "__main__":
  main()
