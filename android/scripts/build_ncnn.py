#!/usr/bin/env python3
"""Build pinned diagnostic Vulkan engine with explicit FP16 cast boundaries."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from prepare_accelerators import SOURCES, prepare_source

ROOT = Path(__file__).resolve().parents[2]


def fresh_source_member(member, destination):
  """Invalidate generated shader/object caches even when a patch is removed."""
  filtered = tarfile.data_filter(member, destination)
  return filtered.replace(mtime=None) if filtered is not None else None


def main():
  native = ROOT / "android/build/native/accelerators"
  prepare_source("ncnn", native)
  glslang = prepare_source("glslang", native)
  patch = ROOT / "android/patches/ncnn-explicit-casts.patch"
  profile_patch = ROOT / "android/patches/ncnn-vulkan-profile.patch"
  tiling_patch = ROOT / "android/patches/ncnn-vulkan-tiling.patch"
  exact_product_patch = ROOT / "android/patches/ncnn-exact-product.patch"
  dispatch_patch = ROOT / "android/patches/ncnn-dispatch-batching.patch"
  normalization_patch = ROOT / "android/patches/ncnn-u8-normalization.patch"
  spatial_patch = ROOT / "android/patches/ncnn-spatial-weight-reuse.patch"
  identity = {"source": SOURCES["ncnn"][1], "archive_sha256": SOURCES["ncnn"][2],
              "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
              "profile_patch_sha256": hashlib.sha256(profile_patch.read_bytes()).hexdigest(),
              "tiling_patch_sha256": hashlib.sha256(tiling_patch.read_bytes()).hexdigest(),
              "exact_product_patch_sha256": hashlib.sha256(exact_product_patch.read_bytes()).hexdigest(),
              "dispatch_patch_sha256": hashlib.sha256(dispatch_patch.read_bytes()).hexdigest(),
              "normalization_patch_sha256": hashlib.sha256(normalization_patch.read_bytes()).hexdigest(),
              "spatial_patch_sha256": hashlib.sha256(spatial_patch.read_bytes()).hexdigest(),
              "recipe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "glslang": SOURCES["glslang"][1], "glslang_archive_sha256": SOURCES["glslang"][2],
              "ndk": "28.2.13676358", "api": 35, "explicit_half_rounding": True,
              "reference_layernorm": 1, "reference_float_gemm": 1, "reference_matmul_rounding": 1}
  target = native / "ncnn-patched"
  stamp = target / "build.json"
  if stamp.exists() and json.loads(stamp.read_text()) == identity:
    return
  target.mkdir(exist_ok=True)
  source = target / ("ncnn-" + identity["source"])
  link = source / "glslang"
  # Archive extraction overwrites tracked sources but leaves patch-created
  # files behind. Remove the disposable export so patch cannot reverse them.
  if source.is_symlink():
    raise ValueError("Unexpected ncnn source link")
  if source.exists():
    shutil.rmtree(source)
  with tarfile.open(native / "ncnn-source.tar.gz") as archive:
    archive.extractall(target, filter=fresh_source_member)
  subprocess.run(["patch", "-f", "-p1", "-i", str(patch)], cwd=source, check=True)
  subprocess.run(["patch", "-f", "-p1", "-i", str(profile_patch)], cwd=source, check=True)
  subprocess.run(["patch", "-f", "-p1", "-i", str(tiling_patch)], cwd=source, check=True)
  subprocess.run(["patch", "-f", "-p1", "-i", str(exact_product_patch)], cwd=source, check=True)
  subprocess.run(["patch", "-f", "-p1", "-i", str(dispatch_patch)], cwd=source, check=True)
  subprocess.run(["patch", "-f", "-p1", "-i", str(normalization_patch)], cwd=source, check=True)
  subprocess.run(["patch", "-f", "-p1", "-i", str(spatial_patch)], cwd=source, check=True)
  if not link.exists():
    if link.is_dir():
      link.rmdir()
    link.symlink_to(glslang, target_is_directory=True)
  elif link.is_dir() and not any(link.iterdir()):
    link.rmdir()
    link.symlink_to(glslang, target_is_directory=True)
  sdk = Path.home() / "Library/Android/sdk"
  cmake = sdk / "cmake/3.31.6/bin/cmake"
  build = target / "android"
  subprocess.run([str(cmake), "-S", str(source), "-B", str(build), "-GNinja",
                  f"-DCMAKE_MAKE_PROGRAM={sdk / 'cmake/3.31.6/bin/ninja'}", "-DCMAKE_BUILD_TYPE=Release",
                  f"-DCMAKE_TOOLCHAIN_FILE={sdk / 'ndk' / identity['ndk'] / 'build/cmake/android.toolchain.cmake'}",
                  "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-35", "-DANDROID_STL=c++_shared",
                  "-DNCNN_SHARED_LIB=ON", "-DNCNN_VULKAN=ON", "-DNCNN_OPENMP=OFF",
                  "-DNCNN_BENCHMARK=ON",
                  "-DNCNN_BUILD_TOOLS=OFF", "-DNCNN_BUILD_EXAMPLES=OFF", "-DNCNN_BUILD_BENCHMARK=OFF",
                  "-DNCNN_BUILD_TESTS=OFF", "-DNCNN_INSTALL_SDK=ON"], check=True)
  subprocess.run([str(cmake), "--build", str(build), "--target", "ncnn", "-j", "4"], check=True)
  subprocess.run([str(cmake), "--install", str(build), "--prefix", str(target / "install")], check=True)
  stamp.write_text(json.dumps(identity, indent=2) + "\n")


if __name__ == "__main__":
  main()
