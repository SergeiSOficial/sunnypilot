#!/usr/bin/env python3
"""Prepare pinned Android native sources without modifying their submodules."""
from pathlib import Path
import io
import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
import re

ROOT = Path(__file__).resolve().parents[1]
LIBUSB_COMMIT = "d52e355daa09f17ce64819122cb067b8a2ee0d4b"  # v1.0.27
MSGQ_COMMIT = "e7396e76dadbb49e374d4b664ff6bbb43a39bcb0"
PYTHON_TARGET = "3.12.15-0"
PYTHON_SHA256 = "bcc3cb4edcff25c8c3103cf45ea35b66ef0a728bb2b3e7731d3c045093ff6d39"


def source_tar(name, url, sha256):
  archive_path = ROOT / "build/native" / name
  if not archive_path.exists():
    with urllib.request.urlopen(url, timeout=60) as response:
      archive_path.write_bytes(response.read())
  with archive_path.open("rb") as source:
    if hashlib.file_digest(source, "sha256").hexdigest() != sha256:
      raise SystemExit(f"{name}: checksum mismatch")
  with tarfile.open(archive_path) as archive:
    archive.extractall(archive_path.parent, filter="data")


def main():
  subprocess.run([sys.executable, str(ROOT / "scripts/stage_panda.py")], check=True)
  # The Android ORT build is a separately validated backend, never a replacement
  # for a failed OpenCL parity check. Pin headers/notices to the matching AAR;
  # compile the library without SME/SVE, which the dedicated AVD cannot execute.
  ort_archive = ROOT / "build/native/onnxruntime-android-1.31.1.aar"
  if not ort_archive.exists():
    ort_archive.parent.mkdir(parents=True, exist_ok=True)
    url = "https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.31.1/onnxruntime-android-1.31.1.aar"
    with urllib.request.urlopen(url, timeout=60) as response:
      ort_archive.write_bytes(response.read())
  if hashlib.sha256(ort_archive.read_bytes()).hexdigest() != "d4d50660300ed2c4f38ffdfd5ab5bfa3a22f995f282dda24f42ea3ea55467b49":
    raise SystemExit("ONNX Runtime Android archive checksum mismatch")
  with zipfile.ZipFile(ort_archive) as archive:
    for name in archive.namelist():
      if (name.startswith("headers/") and name.endswith(".h") and ".." not in Path(name).parts) or name == "jni/arm64-v8a/libonnxruntime.so":
        archive.extract(name, ROOT / "build/native/onnxruntime-1.31.1")
  for name, digest in (("LICENSE", "2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c"),
                       ("ThirdPartyNotices.txt", "143764b952fdb1a7c69ce653bfba74a7744d6a8a573bfb73e235fba356c83de3")):
    notice = ROOT / "build/native/onnxruntime-1.31.1" / name
    if not notice.exists():
      with urllib.request.urlopen(f"https://raw.githubusercontent.com/microsoft/onnxruntime/v1.31.1/{name}", timeout=60) as response:
        notice.write_bytes(response.read())
    if hashlib.sha256(notice.read_bytes()).hexdigest() != digest:
      raise SystemExit(f"ONNX Runtime {name} checksum mismatch")
  subprocess.run([sys.executable, str(ROOT / "scripts/build_ort.py")], check=True)
  source = ROOT / "build/native/libusb"
  if not source.exists():
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "v1.0.27", "https://github.com/libusb/libusb.git", str(source)], check=True)
  actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
  dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
  if actual != LIBUSB_COMMIT or dirty:
    raise SystemExit("libusb checkout differs from its pin; preserve and inspect it before building")
  print(f"libusb {actual}: clean")
  for name, url, commit in (
    ("json11", "https://github.com/dropbox/json11.git", "db00e9369a92aa74bf630a2ffb092a4b0b132c01"),
    ("raylib-python-cffi", "https://github.com/commaai/raylib-python-cffi.git", "a32e9109c9924abdaae5e6e3d5408e210f939ee1"),
  ):
    checkout = ROOT / "build/native" / name
    if not checkout.exists():
      subprocess.run(["git", "clone", url, str(checkout)], check=True)
      subprocess.run(["git", "-C", str(checkout), "checkout", commit], check=True)
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != commit or subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True):
      raise SystemExit(f"{name}: checkout differs from pin")
    if name == "raylib-python-cffi":
      subprocess.run(["git", "-C", str(checkout), "submodule", "update", "--init", "--recursive"], check=True)
      # Patch an exported copy, keeping the pinned source checkout pristine.
      raylib = ROOT / "build/native/raylib-android"
      if raylib.exists():
        shutil.rmtree(raylib)
      raylib.mkdir()
      archive = subprocess.check_output(["git", "-C", str(checkout / "raylib-c"), "archive", "HEAD"])
      with tarfile.open(fileobj=io.BytesIO(archive)) as files:
        files.extractall(raylib, filter="data")
      subprocess.run(["git", "apply", "--directory=android/build/native/raylib-android", "android/patches/raylib-android.patch"],
                     cwd=ROOT.parent, check=True)
  msgq = ROOT / "build/native/msgq"
  source_repo = ROOT.parent / "msgq_repo"
  actual = subprocess.check_output(["git", "-C", str(source_repo), "rev-parse", "HEAD"], text=True).strip()
  if actual != MSGQ_COMMIT:
    raise SystemExit("msgq submodule is not at the Android recipe's pin")
  archive = subprocess.check_output(["git", "-C", str(source_repo), "archive", MSGQ_COMMIT])
  if msgq.exists():
    shutil.rmtree(msgq)
  msgq.mkdir(parents=True)
  with tarfile.open(fileobj=io.BytesIO(archive)) as files:
    files.extractall(msgq, filter="data")
  subprocess.run(["git", "apply", "--directory=android/build/native/msgq", "android/patches/msgq-android.patch"], cwd=ROOT.parent, check=True)
  print(f"msgq {actual}: Android patch applied to generated source")
  # The official Chaquopy target supplies the exact Android pyconfig.h and ABI.
  # No host Python headers or Linux extension binaries enter the APK.
  target = ROOT / f"build/native/target-{PYTHON_TARGET}-arm64-v8a.zip"
  if not target.exists():
    url = f"https://repo.maven.apache.org/maven2/com/chaquo/python/target/{PYTHON_TARGET}/target-{PYTHON_TARGET}-arm64-v8a.zip"
    with urllib.request.urlopen(url, timeout=90) as response:
      target.write_bytes(response.read())
  with target.open("rb") as source:
    if hashlib.file_digest(source, "sha256").hexdigest() != PYTHON_SHA256:
      raise SystemExit("Chaquopy target checksum mismatch")
  with zipfile.ZipFile(target) as archive:
    for name in archive.namelist():
      if name.startswith("include/") or name == "jniLibs/arm64-v8a/libpython3.12.so":
        archive.extract(name, ROOT / "build/native/python")
  for module in ("msgq.ipc_pyx", "msgq.visionipc.visionipc_pyx"):
    path = msgq / (module.replace(".", "/") + ".pyx")
    subprocess.run([sys.executable, "-m", "cython", "--cplus", "-3", "-I", str(msgq),
                    "--module-name", module, "-o", str(path.with_suffix(".cpp")), str(path)], check=True)
  print(f"Python {PYTHON_TARGET}: verified Android headers; Cython sources generated")
  # Match pyproject.toml's pycapnp version and its bundled Cap'n Proto release.
  source_tar("pycapnp-2.1.0.tar.gz",
             "https://files.pythonhosted.org/packages/15/86/a57e3c92acd3e1d2fc3dcad683ada191f722e4ac927e1a384b228ec2780a/pycapnp-2.1.0.tar.gz",
             "69cc3d861fee1c9b26c73ad2e8a5d51e76ad87e4ff9be33a4fd2fc72f5846aec")
  source_tar("capnproto-c++-1.0.1.tar.gz", "https://capnproto.org/capnproto-c++-1.0.1.tar.gz",
             "0f7f4b8a76a2cdb284fddef20de8306450df6dd031a47a15ac95bc43c3358e09")
  subprocess.run(["git", "apply", "--directory=android/build/native/capnproto-c++-1.0.1",
                  "android/patches/capnp-android.patch"], cwd=ROOT.parent, check=True)
  # Params references schema enums. Generate matching headers explicitly rather
  # than accidentally consuming files from an earlier host SCons build.
  compiler = Path(sys.executable).parent / "capnpc"
  if not compiler.is_file() or subprocess.check_output([str(compiler), "--version"], text=True).strip() != "Cap'n Proto version 1.0.1":
    raise SystemExit("Install the locked capnproto host package (compiler 1.0.1) into this Python environment")
  generated = ROOT / "build/native/generated/openpilot/cereal/gen/cpp"
  generated.mkdir(parents=True, exist_ok=True)
  subprocess.run([str(compiler), "--src-prefix=openpilot/cereal", "--src-prefix=opendbc_repo/opendbc/car",
                  "--import-path=opendbc_repo/opendbc/car", "-o", f"c++:{generated}",
                  *[f"openpilot/cereal/{s}.capnp" for s in ("log", "deprecated", "custom")],
                  "opendbc_repo/opendbc/car/car.capnp"], cwd=ROOT.parent, check=True)
  source_tar("pyzmq-27.2.0.tar.gz",
             "https://files.pythonhosted.org/packages/e7/8d/5b3d5631c2f4b4b8862f64cd0c9eb777b5710eeb5125b4be8dd0a200a4c0/pyzmq-27.2.0.tar.gz",
             "54d4259d1bfae24ecdb5ca79f7acc2eac6c286a02d6a0ae617797cb45f0726d3")
  source_tar("libzmq-4.3.5.tar.gz", "https://codeload.github.com/zeromq/libzmq/tar.gz/refs/tags/v4.3.5",
             "6c972d1e6a91a0ecd79c3236f04cf0126f2f4dfbbad407d72b4606a7ba93f9c6")
  source_tar("zstandard-0.25.0.tar.gz",
             "https://files.pythonhosted.org/packages/fd/aa/3e0508d5a5dd96529cdc5a97011299056e14c6505b678fd58938792794b1/zstandard-0.25.0.tar.gz",
             "7713e1179d162cf5c7906da876ec2ccb9c3a9dcbdffef0cc7f70c3667a205f0b")
  zmq = ROOT / "build/native/pyzmq-27.2.0"
  subprocess.run([sys.executable, "-m", "cython", "--cplus", "-3", "-I", str(zmq),
                  "--module-name", "zmq.backend.cython._zmq", "-o", str(zmq / "_zmq.cpp"),
                  str(zmq / "zmq/backend/cython/_zmq.py")], check=True)
  # Generate the pinned pyray CFFI API without executing a host linker recipe.
  # CMake below links this C code to the Android Raylib instance.
  from cffi import FFI
  ray = ROOT / "build/native/raylib-python-cffi"
  ffi = FFI()
  # Same preprocessing as the pinned binding generator, without overwriting its
  # tracked *.modified files or linking macOS libraries into the target.
  for header in ("raylib.h", "rlgl.h", "raymath.h"):
    source = "".join(line for line in (ray / "raylib-c/src" / header).read_text().splitlines(True)
                     if "#include" not in line and "TraceLogCallback" not in line)
    defines = ["RL_MATRIX_TYPE", "RL_QUATERNION_TYPE", "RL_VECTOR4_TYPE", "RL_VECTOR3_TYPE", "RL_VECTOR2_TYPE",
               "RLAPI=", "PHYSACDEF=", "RAYGUIDEF=", "RMAPI=", "GRAPHICS_API_OPENGL_ES3"]
    source = subprocess.run(["gcc", "-CC", "-P", "-undef", "-nostdinc", *("-D" + d for d in defines), "-dDI", "-E", "-"],
                            input=source, text=True, capture_output=True, check=True).stdout.replace("va_list", "void *")
    if header == "raymath.h":
      source = re.sub(r"\n\{\n(.|\n)*?\n}\n", ";", source)
    ffi.cdef("\n".join(line for line in source.splitlines() if not line.startswith("#")))
  ffi.set_source("raylib._raylib_cffi", '#include "raylib.h"\n#include "rlgl.h"\n#include "raymath.h"\n')
  ffi.emit_c_code(str(ROOT / "build/native/raylib_cffi.c"))
  import tomllib
  lock = tomllib.loads((ROOT.parent / "uv.lock").read_text())
  for package in lock["package"]:
    if package["name"] == "setproctitle":
      sdist = package["sdist"]
      source_tar(f"setproctitle-{package['version']}.tar.gz", sdist["url"], sdist["hash"].removeprefix("sha256:"))


if __name__ == "__main__":
  main()
