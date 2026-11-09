#!/usr/bin/env python3
"""Package the inspected source identity; Android has no runtime Git checkout."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from openpilot.common.version import get_build_metadata


def main():
  metadata = asdict(get_build_metadata(str(ROOT)))
  metadata["openpilot"]["build_style"] = "android-debug-rx-only"
  target = Path(sys.argv[1])
  target.parent.mkdir(parents=True, exist_ok=True)
  target.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
  main()
