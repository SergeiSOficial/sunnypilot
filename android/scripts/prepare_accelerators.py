#!/usr/bin/env python3
"""Fetch hash-pinned upstream diagnostic GPU engines; never change model assets."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = {
  "mnn-android": ("alibaba/MNN", "3.6.2", "mnn_3.6.2_android_armv7_armv8_cpu_opencl_vulkan.zip",
                  "0b4cc255964de1e3ac8dce75dce316699a0a28b914ecacb7bea8465666858ece"),
  "mnn-macos": ("alibaba/MNN", "3.6.2", "mnn_3.6.2_macos_x64_arm82_cpu_opencl_metal.zip",
                "827ad7930e1a13170e6a9bf904f1dbc8c0457c66624e32818ee11494fbf5a0e5"),
  "ncnn-android": ("Tencent/ncnn", "20261020", "ncnn-20261020-android-vulkan.zip",
                   "6e0c5f9b397faf32b923de2ef6210af9bf375851869dceff06a8005331981086"),
  "ncnn-macos": ("Tencent/ncnn", "20261020", "ncnn-20261020-macos.zip",
                 "e33de27e608da9f71c7c47b61cc2c443f08371f1df47a8222ea493ccd93e3f2d"),
}
SOURCES = {
  "glslang": ("nihui/glslang", "780b6a209132dc74e99458f62ebe5fa7e57c41cf", "9c80e985ae0ea5c7b6a0d59b99de3058ba170fcc0589a913023d12c3473a3617"),
  "mnn": ("alibaba/MNN", "7e3c6fb5225d3ccf22807ee641332669df34650d",
          "01dcdce39072b70188079a1dca0600e782c8dd0fcf2ad77f610bab264166a0cf"),
  "ncnn": ("Tencent/ncnn", "7e36d25cb153092ba87a60530455c7376fedf452",
           "5197a2ea772c97e8bc74547fe37f833d494476603a30bd5216371ea0c1751ba1"),
}
TOOLS = {
  "mnn": ("https://files.pythonhosted.org/packages/0a/ca/9c2876dada20a5e01224950fcac59efe52e58780ca11f7d20938dbf3ce06/mnn-3.6.2-cp312-cp312-macosx_11_0_arm64.whl",
          "4a378db68122066bf31730d97b71e48dc7f7b1a9030da616b4b42f1e93317b8c"),
  "pnnx": ("https://files.pythonhosted.org/packages/8c/e7/98373b217be926b9a6895c9316c350f97fad8bedeb64e5f43cc99c0f498e/pnnx-20261020-py3-none-macosx_10_9_universal2.macosx_10_9_x86_64.macosx_11_0_arm64.whl",
           "f4d60b7df5d1f7329f60cf75018ab4e25006db6d2299cf5ef8d56c051044660b"),
}


def prepare_tool(engine, destination):
  url, expected = TOOLS[engine]
  archive = destination / url.rsplit("/", 1)[1]
  destination.mkdir(parents=True, exist_ok=True)
  if not archive.exists():
    urllib.request.urlretrieve(url, archive)
  with archive.open("rb") as source:
    if hashlib.file_digest(source, "sha256").hexdigest() != expected:
      raise ValueError(f"Converter wheel hash mismatch: {engine}")
  target = destination / ("mnn-python" if engine == "mnn" else "pnnx")
  with zipfile.ZipFile(archive) as source:
    for entry in source.infolist():
      if not (target / entry.filename).resolve().is_relative_to(target.resolve()):
        raise ValueError("Converter archive path escapes destination")
    source.extractall(target)
  if engine == "mnn":
    (target / "lib").mkdir(exist_ok=True)
    for library in target.glob("*.data/data/lib/*.dylib"):
      shutil.copy2(library, target / "lib" / library.name)
  else:
    (target / "pnnx/pnnx").chmod(0o755)
  return target


def prepare_source(engine, destination):
  repo, commit, expected = SOURCES[engine]
  destination.mkdir(parents=True, exist_ok=True)
  archive = destination / f"{engine}-source.tar.gz"
  if not archive.exists():
    urllib.request.urlretrieve(f"https://codeload.github.com/{repo}/tar.gz/{commit}", archive)
  with archive.open("rb") as source:
    actual = hashlib.file_digest(source, "sha256").hexdigest()
  if actual != expected:
    raise ValueError(f"Source archive hash mismatch: {engine}")
  target = destination / f"{engine}-source"
  folder = target / f"{repo.split('/')[-1]}-{commit}"
  if not folder.exists():
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as source:
      source.extractall(target, filter="data")
  return folder


def prepare(name, destination):
  repo, version, filename, expected = ARTIFACTS[name]
  url = f"https://github.com/{repo}/releases/download/{version}/{filename}"
  destination.mkdir(parents=True, exist_ok=True)
  archive = destination / filename
  if not archive.exists():
    temporary = archive.with_suffix(".download")
    urllib.request.urlretrieve(url, temporary)
    temporary.rename(archive)
  with archive.open("rb") as source:
    actual = hashlib.file_digest(source, "sha256").hexdigest()
  if actual != expected:
    raise ValueError(f"Upstream archive hash mismatch: {filename}")
  extracted = destination / name
  if not (extracted / "artifact.json").exists():
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
      for entry in source.infolist():
        path = (extracted / entry.filename).resolve()
        if not path.is_relative_to(extracted.resolve()):
          raise ValueError("Archive path escapes destination")
      source.extractall(extracted)
    (extracted / "artifact.json").write_text(json.dumps({"version": version, "url": url, "sha256": actual}, indent=2) + "\n")
  return extracted


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--engine", choices=ARTIFACTS, action="append")
  parser.add_argument("--destination", type=Path, default=ROOT / "android/build/native/accelerators")
  parser.add_argument("--sources", action="store_true")
  parser.add_argument("--tools", action="store_true", help="Matching local converters for macOS ARM64 Python 3.12")
  args = parser.parse_args()
  for engine in args.engine or ARTIFACTS:
    print(engine, prepare(engine, args.destination), flush=True)
  if args.sources:
    for engine in SOURCES:
      print(engine, prepare_source(engine, args.destination), flush=True)
  if args.tools:
    for engine in TOOLS:
      print(engine, prepare_tool(engine, args.destination), flush=True)
