#!/usr/bin/env python3
"""Build pinned MNN with explicit half casts, preserving the release baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

from prepare_accelerators import SOURCES, prepare_source

ROOT = Path(__file__).resolve().parents[2]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--profile-gpu", action="store_true", help="Build diagnostic event-profiling libraries separately from APK libraries")
  args = parser.parse_args()
  native = ROOT / "android/build/native/accelerators"
  prepare_source("mnn", native)  # verifies the pinned source archive
  patch = ROOT / "android/patches/mnn-half-cast.patch"
  identity = {"source": SOURCES["mnn"][1], "archive_sha256": SOURCES["mnn"][2],
              "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
              "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "ndk": "28.2.13676358", "api": 35, "gpu_profiling": args.profile_gpu,
              "profile_output": "stdout" if args.profile_gpu else "logcat"}
  target = native / "mnn-patched"
  stamp = target / ("build-profile.json" if args.profile_gpu else "build.json")
  if stamp.exists() and json.loads(stamp.read_text()) == identity:
    return
  prepared = target / "source.json"
  source_identity = {key: identity[key] for key in ("source", "archive_sha256", "patch_sha256")}
  previous = json.loads((prepared if prepared.exists() else stamp).read_text()) if prepared.exists() or stamp.exists() else {}
  source_changed = any(previous.get(key) != value for key, value in source_identity.items())
  if source_changed:
    if target.exists():
      if any(previous.get(key) != identity[key] for key in ("source", "archive_sha256")):
        shutil.rmtree(target)
      else:
        # Re-extract pristine pinned sources, retaining CMake's object cache.
        # Archive timestamps leave untouched translation units up to date.
        shutil.rmtree(target / ("MNN-" + identity["source"]), ignore_errors=True)
    target.mkdir(exist_ok=True)
    with tarfile.open(native / "mnn-source.tar.gz") as archive:
      archive.extractall(target, filter="data")
    source = target / ("MNN-" + identity["source"])
    subprocess.run(["patch", "-p1", "-i", str(patch)], cwd=source, check=True)
    prepared.write_text(json.dumps(source_identity, indent=2) + "\n")
  source = target / ("MNN-" + identity["source"])
  kernels = source / "source/backend/opencl/execution/cl"
  if source_changed:
    subprocess.run([sys.executable, str(kernels / "opencl_codegen.py"), str(kernels)],
                   cwd=kernels, check=True, stdout=subprocess.DEVNULL)
  sdk = Path(os.environ.get("ANDROID_HOME", Path.home() / "Library/Android/sdk"))
  cmake = sdk / "cmake/3.31.6/bin/cmake"
  ninja = sdk / "cmake/3.31.6/bin/ninja"
  common = ["-GNinja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-DCMAKE_BUILD_TYPE=Release",
            "-DMNN_BUILD_DEMO=OFF", "-DMNN_BUILD_TOOLS=OFF", "-DMNN_BUILD_TEST=OFF",
            "-DMNN_BUILD_TRAIN=OFF", "-DMNN_BUILD_BENCHMARK=OFF", "-DMNN_VULKAN=OFF",
            "-DMNN_OPENMP=OFF", "-DMNN_AAPL_FMWK=OFF", "-DMNN_SEP_BUILD=ON"]
  host = target / "host"
  subprocess.run([str(cmake), "-S", str(source), "-B", str(host), *common,
                  "-DMNN_BUILD_CONVERTER=ON", "-DMNN_METAL=OFF", "-DMNN_OPENCL=OFF"], check=True)
  subprocess.run([str(cmake), "--build", str(host), "--target", "MNNConvert", "-j", "4"], check=True)
  android = target / ("android-profile" if args.profile_gpu else "android")
  subprocess.run([str(cmake), "-S", str(source), "-B", str(android), *common,
                  f"-DCMAKE_TOOLCHAIN_FILE={sdk / 'ndk' / identity['ndk'] / 'build/cmake/android.toolchain.cmake'}",
                  "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-35",
                  "-DANDROID_STL=c++_shared", f"-DNATIVE_LIBRARY_OUTPUT:PATH={android / 'libs'}",
                  "-DMNN_BUILD_CONVERTER=OFF", "-DMNN_OPENCL=ON",
                  f"-DMNN_GPU_TIME_PROFILE={'ON' if args.profile_gpu else 'OFF'}", "-DMNN_GPU_PROFILE_SILENT=OFF",
                  f"-DMNN_USE_LOGCAT={'OFF' if args.profile_gpu else 'ON'}"], check=True)
  subprocess.run([str(cmake), "--build", str(android), "--target", "MNN", "MNN_CL", "-j", "4"], check=True)
  stamp.write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
  main()
