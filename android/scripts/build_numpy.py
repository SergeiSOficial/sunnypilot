#!/usr/bin/env python3
"""Build pinned NumPy for Chaquopy's Bionic/Python ABI, never a Linux wheel."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
VERSION = "2.5.3"
SHA256 = "df2d5874ff183595a4ba404edd04f6bd9b5505c1d7708573f6a6c17489a67563"


def main():
  native = ROOT / "android/build/native"
  archive = native / f"numpy-{VERSION}.tar.gz"
  if not archive.exists():
    with urllib.request.urlopen(f"https://files.pythonhosted.org/packages/source/n/numpy/numpy-{VERSION}.tar.gz", timeout=60) as response:
      archive.write_bytes(response.read())
  if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
    raise ValueError("NumPy source checksum mismatch")
  sdk = Path(os.environ.get("ANDROID_HOME", Path.home() / "Library/Android/sdk"))
  toolchain = sdk / "ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin"
  python_root = native / "python"
  include = python_root / "include/python3.12"
  if not (include / "Python.h").exists():
    raise RuntimeError("Run prepare_native.py first to extract the pinned Chaquopy Python target")
  source, build, install = (native / name for name in (f"numpy-{VERSION}", "numpy-build", "numpy-install"))
  signature = {"source": SHA256, "python": "3.12.15-0", "ndk": "28.2.13676358", "api": 35,
               "recipe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  stamp = install / "build.json"
  if stamp.exists() and json.loads(stamp.read_text()) == signature:
    return
  if source.exists():
    shutil.rmtree(source)
  with tarfile.open(archive) as files:
    files.extractall(native, filter="data")
  # Meson's build Python runs on macOS; extension headers and linkage must use
  # the separately pinned Android target instead of that interpreter's ABI.
  meson_file = source / "meson.build"
  text = meson_file.read_text()
  original = "py_dep = py.dependency()"
  if text.count(original) != 1:
    raise ValueError("NumPy's Python dependency recipe changed")
  meson_file.write_text(text.replace(original,
    f"py_dep = declare_dependency(include_directories: include_directories('{include}'), "
    f"link_args: ['-L{python_root / 'jniLibs/arm64-v8a'}', '-lpython3.12'])"))
  cross = native / "numpy-android.ini"
  cross.write_text(f"""[binaries]
c = '{toolchain / 'aarch64-linux-android35-clang'}'
cpp = '{toolchain / 'aarch64-linux-android35-clang++'}'
ar = '{toolchain / 'llvm-ar'}'
strip = '{toolchain / 'llvm-strip'}'
pkg-config = 'false'
[host_machine]
system = 'android'
cpu_family = 'aarch64'
cpu = 'aarch64'
endian = 'little'
[properties]
needs_exe_wrapper = true
longdouble_format = 'IEEE_QUAD_LE'
[built-in options]
c_args = ['-O2']
cpp_args = ['-O2']
c_link_args = ['-L{python_root / 'jniLibs/arm64-v8a'}', '-Wl,--no-as-needed', '-lpython3.12', '-Wl,--as-needed']
cpp_link_args = ['-L{python_root / 'jniLibs/arm64-v8a'}', '-Wl,--no-as-needed', '-lpython3.12', '-Wl,--as-needed']
""")
  for directory in (build, install):
    if directory.exists():
      shutil.rmtree(directory)
  env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
  env["PYTHONPATH"] = str(source / "vendored-meson/meson")
  subprocess.run([sys.executable, "-m", "mesonbuild.mesonmain", "setup", str(build), str(source),
                  f"--cross-file={cross}", "--prefix=/usr", "-Dblas=none", "-Dlapack=none", "-Dallow-noblas=true",
                  "-Dcpu-baseline=neon", "-Dcpu-dispatch=none", "-Dbuildtype=release"], env=env, check=True)
  subprocess.run([sys.executable, "-m", "mesonbuild.mesonmain", "compile", "-C", str(build), "-j", "4"], env=env, check=True)
  subprocess.run([sys.executable, "-m", "mesonbuild.mesonmain", "install", "-C", str(build), f"--destdir={install}"], env=env, check=True)
  packages = list(install.rglob("numpy/_core/_multiarray_umath.*.so"))
  if len(packages) != 1:
    raise ValueError("NumPy install layout changed")
  package = packages[0].parents[2]
  for extension in package.rglob("*.so"):
    dynamic = subprocess.check_output([str(toolchain / "llvm-readelf"), "-d", str(extension)], text=True)
    if "Shared library: [libpython3.12.so]" not in dynamic:
      raise ValueError(f"Android NumPy extension does not link the target Python runtime: {extension.name}")
    # The build Python determines the filename suffix, not the object ABI.
    # Android's extension loader accepts .so without a host-specific suffix.
    extension.rename(extension.with_name(extension.name.split(".")[0] + ".so"))
  (install / "package-path.txt").write_text(str(package))
  stamp.write_text(json.dumps(signature, indent=2))


if __name__ == "__main__":
  main()
