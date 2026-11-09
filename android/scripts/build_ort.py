#!/usr/bin/env python3
"""Build pinned Android ORT without optional instructions the API 35 AVD traps."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "6af1a932e2286d5c58166085d24c68783033edee"
NDK_VERSION = "28.2.13676358"
CMAKE_VERSION = "3.31.6"


def digest(path):
  with path.open("rb") as source:
    return hashlib.file_digest(source, "sha256").hexdigest()


def main():
  native = ROOT / "build/native"
  native.mkdir(parents=True, exist_ok=True)
  source = native / "onnxruntime-source-1.31.1"
  if not source.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--depth", "1", "--branch", "v1.31.1",
                    "https://github.com/microsoft/onnxruntime.git", str(source)], check=True)
  def git(*args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
  if git("rev-parse", "HEAD") != COMMIT:
    raise SystemExit("ORT source differs from its pin; preserve and inspect it before building")
  patch = ROOT / "patches/onnxruntime-fp16-groups.patch"
  if not git("status", "--porcelain"):
    subprocess.run(["git", "-C", str(source), "apply", "--check", str(patch)], check=True)
    subprocess.run(["git", "-C", str(source), "apply", str(patch)], check=True)
  if (git("diff", "--binary", "HEAD") != patch.read_text().strip() or
      git("diff", "--cached", "--name-only") or git("ls-files", "--others", "--exclude-standard")):
    raise SystemExit("ORT source contains changes beyond the reproducible FP16 groups patch")
  subprocess.run(["git", "-C", str(source), "submodule", "update", "--init", "--depth", "1", "cmake/external/onnx"], check=True)
  onnx_commit = git("ls-tree", "HEAD", "cmake/external/onnx").split()[2]
  actual = subprocess.check_output(["git", "-C", str(source / "cmake/external/onnx"), "rev-parse", "HEAD"], text=True).strip()
  if actual != onnx_commit:
    raise SystemExit("ORT ONNX submodule pin mismatch")
  sdk = Path(os.environ.get("ANDROID_HOME", str(Path.home() / "Library/Android/sdk")))
  build = native / "onnxruntime-portable"
  command = [sys.executable, str(source / "tools/ci_build/build.py"), "--build_dir", str(build), "--config", "Release",
             "--update", "--build", "--build_shared_lib", "--parallel", "4", "--skip_tests", "--skip_submodule_sync",
             "--android", "--android_sdk_path", str(sdk), "--android_ndk_path", str(sdk / "ndk" / NDK_VERSION),
             "--android_abi", "arm64-v8a", "--android_api", "28", "--android_cpp_shared", "--use_nnapi",
             "--no_kleidiai", "--no_sve", "--no_telemetry", "--cmake_path", str(sdk / "cmake" / CMAKE_VERSION / "bin/cmake"),
             # Eigen::half changes its calling convention when FP16 is enabled.
             # Every translation unit must agree; per-source flags break the ABI.
             "--cmake_generator", "Ninja", "--cmake_extra_defines", "onnxruntime_BUILD_UNIT_TESTS=OFF",
             "CMAKE_CXX_FLAGS=-march=armv8-a+fp16"]
  subprocess.run(command, cwd=ROOT.parent, check=True)
  library = build / "Release/libonnxruntime.so"
  destination = native / "onnxruntime-1.31.1/jni/arm64-v8a/libonnxruntime.so"
  destination.parent.mkdir(parents=True, exist_ok=True)
  shutil.copy2(library, destination)
  manifest = {"commit": COMMIT, "onnx_commit": onnx_commit, "ndk": NDK_VERSION, "cmake": CMAKE_VERSION,
              "requires_arm64_fp16": True,
              "optional_sme_sve_kernels": False, "patch_sha256": digest(patch),
              "library_sha256": digest(destination), "command": command,
              "vehicle_tx_enabled": False}
  (destination.parents[2] / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
  print(json.dumps({key: value for key, value in manifest.items() if key != "command"}, indent=2))


if __name__ == "__main__":
  main()
