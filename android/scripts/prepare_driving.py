#!/usr/bin/env python3
"""Generate pinned Kalman/MPC sources and cross-build their Android dependencies."""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "android/build/native"
ACADOS_PIN = "8af9b0ad180940ef611884574a0b27a43504311d"
REDNOSE_PIN = "28d4a7f69e80e1c3e0d24ca0733d7daeaeade3d0"


def run(*command, **kwargs):
  subprocess.run([str(x) for x in command], check=True, **kwargs)


def checkout(directory, url, pin):
  if not directory.exists():
    run("git", "clone", "--no-checkout", url, directory)
    run("git", "-C", directory, "checkout", pin)
  actual = subprocess.check_output(["git", "-C", str(directory), "rev-parse", "HEAD"], text=True).strip()
  dirty = subprocess.check_output(["git", "-C", str(directory), "status", "--porcelain"], text=True).strip()
  if actual != pin or dirty:
    raise ValueError(f"{directory.name}: pinned sources changed; inspect before rebuilding")
  run("git", "-C", directory, "submodule", "update", "--init", "--recursive")
  states = subprocess.check_output(["git", "-C", str(directory), "submodule", "status", "--recursive"], text=True)
  if any(line[0] != " " for line in states.splitlines()):
    raise ValueError("acados submodule pins differ")
  return states


def main():
  import acados
  import casadi
  import eigen
  dependencies = {name: importlib.metadata.version(name) for name in ("comma-deps-eigen", "comma-deps-acados")}
  if dependencies != {"comma-deps-eigen": "3.4.0.post103", "comma-deps-acados": "0.2.2.post103"}:
    raise ValueError(f"Code generation dependencies differ from repository lock: {dependencies}")
  if casadi.__version__ != "3.6.7":
    raise ValueError("Code generation requires the pinned CasADi 3.6.7")
  source = NATIVE / "acados-src"
  submodules = checkout(source, "https://github.com/acados/acados.git", ACADOS_PIN)
  sdk = Path(os.environ.get("ANDROID_HOME", Path.home() / "Library/Android/sdk"))
  cmake = sdk / "cmake/3.31.6/bin/cmake"
  install, build = NATIVE / "acados-install", NATIVE / "acados-build"
  signature = {"pin": ACADOS_PIN, "submodules": submodules, "ndk": "28.2.13676358", "api": 35,
               "blasfeo": "GENERIC", "hpipm": "GENERIC", "shared": False,
               "recipe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  stamp = install / "build.json"
  if not stamp.exists() or json.loads(stamp.read_text()) != signature:
    run(cmake, "-S", source, "-B", build,
        f"-DCMAKE_TOOLCHAIN_FILE={sdk}/ndk/28.2.13676358/build/cmake/android.toolchain.cmake",
        "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-35", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON", f"-DACADOS_INSTALL_DIR={install}",
        "-DBLASFEO_TARGET=GENERIC", "-DHPIPM_TARGET=GENERIC", "-DBUILD_SHARED_LIBS=OFF",
        "-DACADOS_WITH_QPOASES=OFF", "-DACADOS_EXAMPLES=OFF", "-DACADOS_UNIT_TESTS=OFF")
    run(cmake, "--build", build, "--parallel", "4")
    run(cmake, "--install", build)
    stamp.write_text(json.dumps(signature, indent=2))

  rednose = ROOT / "rednose_repo"
  if subprocess.check_output(["git", "-C", str(rednose), "rev-parse", "HEAD"], text=True).strip() != REDNOSE_PIN:
    raise ValueError("Rednose pin changed")
  export = NATIVE / "driving"
  export.mkdir(exist_ok=True)
  # Use only tracked sources; target extensions are rebuilt with NumPy 2 headers.
  rednose_export = export / "rednose-src"
  if rednose_export.exists():
    shutil.rmtree(rednose_export)
  rednose_export.mkdir()
  archive = subprocess.check_output(["git", "-C", str(rednose), "archive", REDNOSE_PIN])
  with tarfile.open(fileobj=io.BytesIO(archive)) as files:
    files.extractall(rednose_export, filter="data")
  run(sys.executable, "-m", "cython", "--cplus", "-3", "-I", rednose_export,
      "--module-name", "rednose.helpers.ekf_sym_pyx", "-o", export / "ekf_sym_pyx.cpp",
      rednose_export / "rednose/helpers/ekf_sym_pyx.pyx")
  generated = export / "kalman"
  generated.mkdir(exist_ok=True)
  environment = {**os.environ, "PYTHONPATH": str(ROOT), "ACADOS_SOURCE_DIR": str(source),
                 "TERA_PATH": str(acados.TERA_PATH)}
  for model in ("pose", "car"):
    run(sys.executable, ROOT / f"openpilot/selfdrive/locationd/models/{model}_kf.py", "ignored", generated, env=environment)

  mpc = export / "mpc"
  mpc.mkdir(exist_ok=True)
  shutil.copy2(ROOT / "openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py", mpc / "long_mpc.py")
  run(sys.executable, mpc / "long_mpc.py", cwd=mpc, env=environment)
  template = Path(acados.TEMPLATE_DIR)
  run(sys.executable, "-m", "cython", "-3", "-I", mpc / "c_generated_code", "-I", template,
      "--module-name", "acados_ocp_solver_pyx", "-o", mpc / "acados_ocp_solver_pyx.c",
      template / "acados_ocp_solver_pyx.pyx")
  eigen_headers = Path(eigen.__file__).parent / "install/eigen3"
  shutil.copytree(eigen_headers, export / "include/eigen3", dirs_exist_ok=True)
  (export / "build.json").write_text(json.dumps({"acados": signature, "rednose": REDNOSE_PIN,
    "numpy": "2.5.3", "casadi": casadi.__version__, "codegen_dependencies": dependencies,
    "eigen_headers_sha256": hashlib.sha256(b"".join(
      str(path.relative_to(eigen_headers)).encode() + b"\0" + path.read_bytes() for path in sorted(eigen_headers.rglob("*")) if path.is_file()
    )).hexdigest(), "source_only": True}, indent=2))


if __name__ == "__main__":
  main()
