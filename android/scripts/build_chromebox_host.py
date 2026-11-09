#!/usr/bin/env python3
"""Rebuild Linux IPC/parameter bindings without changing the Android artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from convert_snpe_linux import digest


def build(root):
  repo = root/'repo'; native = repo/'android/build/native'
  python = root/'host-venv/bin/python'; binary = root/'tools/bin'
  env = dict(os.environ, PATH=str(root/'host-venv/bin')+':'+str(binary)+':'+os.environ['PATH'])
  uv = root/'snpe-tools/uv-x86_64-unknown-linux-gnu/uv'
  for package in ('opendbc_repo', 'msgq_repo', 'tinygrad_repo'):
    subprocess.run([str(uv), 'pip', 'install', '--python', str(python), '--no-deps', '-e', str(repo/package)], check=True)
  subprocess.run([str(root/'host-venv/bin/scons'), '--minimal', '-j2',
    'CC='+str(binary/'gcc'), 'CXX='+str(binary/'g++')], cwd=repo/'msgq_repo', env=env, check=True)
  import_site = subprocess.check_output([str(python), '-c', 'import sysconfig;print(sysconfig.get_path("platlib"))'], text=True).strip()
  libraries = list((Path(import_site)/'pyzmq.libs').glob('libzmq*.so*'))
  if len(libraries) != 1: raise ValueError('Ambiguous pinned pyzmq library')
  includes = [repo/'openpilot', repo, native, native/'generated', native/'generated/openpilot/cereal/gen/cpp',
              native/'capnproto-c++-1.0.1/src', native/'libzmq-4.3.5/include']
  sources = [repo/'openpilot/common'/name for name in ('params.cc', 'params_c.cc', 'util.cc', 'swaglog.cc')]
  sources.append(native/'json11/json11.cpp')
  target = repo/'openpilot/common/libparams_c.so'
  command = [str(binary/'g++'), '-std=c++17', '-include', 'cstdint', '-O2', '-shared', '-fPIC',
             *['-I'+str(p) for p in includes], *map(str,sources), str(libraries[0]),
             '-Wl,-rpath,'+str(libraries[0].parent), '-Wl,--no-undefined', '-o', str(target)]
  subprocess.run(command, env=env, check=True)
  subprocess.run([str(python), '-c', 'import msgq;from openpilot.common.params import Params;print("Linux IPC and params import pass")'],
                 cwd=repo, env=env, check=True)
  paths = [target, repo/'msgq_repo/msgq/ipc_pyx.so', repo/'msgq_repo/msgq/visionipc/visionipc_pyx.so']
  (root/'host-bindings.json').write_text(json.dumps({'command': command,
    'artifacts': {str(p.relative_to(repo)): digest(p) for p in paths},
    'source_sha256': {str(p.relative_to(repo)): digest(p) for p in sources},
    'vehicle_tx_enabled': False, 'qualified': False}, indent=2)+'\n')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True, type=Path)
  build(p.parse_args().root.resolve())
