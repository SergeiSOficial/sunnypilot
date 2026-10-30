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
  source = ROOT / "build/native/libusb"
  if not source.exists():
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "v1.0.27", "https://github.com/libusb/libusb.git", str(source)], check=True)
  actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
  dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
  if actual != LIBUSB_COMMIT or dirty:
    raise SystemExit("libusb checkout differs from its pin; preserve and inspect it before building")
  print(f"libusb {actual}: clean")
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


if __name__ == "__main__":
  main()
