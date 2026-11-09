#!/usr/bin/env python3
"""Fetch the pinned official Android QNN provider and runtime for opt-in diagnostics."""
import hashlib
from pathlib import Path
import shutil
import urllib.request


DESTINATION = Path(__file__).resolve().parents[1] / "build/native/qnn-2.7.0"
ARTIFACTS = {
  "onnxruntime-android-qnn-2.7.0.aar": (
    "https://repo.maven.apache.org/maven2/com/qualcomm/qti/onnxruntime-android-qnn/2.7.0/onnxruntime-android-qnn-2.7.0.aar",
    "ecd36df222e2a947ceb13bdd6fec181d1f1841b40379e21ce519830f2fbd562a"),
  "qnn-runtime-2.51.0.aar": (
    "https://repo.maven.apache.org/maven2/com/qualcomm/qti/qnn-runtime/2.51.0/qnn-runtime-2.51.0.aar",
    "6388face1d5aa5e8958d745e5a547e92b5bec6fc332dfb1b438b1e17ffa02861"),
}


def digest(path):
  with path.open("rb") as source:
    return hashlib.file_digest(source, "sha256").hexdigest()


def main():
  DESTINATION.mkdir(parents=True, exist_ok=True)
  for name, (url, expected) in ARTIFACTS.items():
    target = DESTINATION / name
    if target.exists():
      if digest(target) != expected:
        raise ValueError(f"Hash mismatch for existing artifact: {target}")
    else:
      temporary = target.with_suffix(".download")
      try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
          shutil.copyfileobj(response, output)
        if digest(temporary) != expected:
          raise ValueError(f"Hash mismatch for downloaded artifact: {name}")
        temporary.replace(target)
      finally:
        temporary.unlink(missing_ok=True)
    print(f"{name}: SHA-256 verified")


if __name__ == "__main__":
  main()
