#!/usr/bin/env python3
"""Build-tested, user-owned Ubuntu native toolchain; no system installation."""
import argparse
import json
from pathlib import Path
import platform
import shlex
import subprocess
from convert_snpe_linux import digest

PACKAGES = ('ripgrep make gcc-15-x86-64-linux-gnu g++-15-x86-64-linux-gnu cpp-15-x86-64-linux-gnu '
  'libgcc-15-dev libstdc++-15-dev binutils-x86-64-linux-gnu libbinutils libctf0 libctf-nobfd0 '
  'libsframe3 libgprofng0 libc6-dev libc6 linux-libc-dev libcrypt-dev libcrypt1 '
  'libisl23 libmpfr6 libmpc3 libgmp10 libgcc-s1 libstdc++6').split()


def install(root):
  if platform.system() != 'Linux' or platform.machine() != 'x86_64': raise RuntimeError('Linux x86_64 only')
  packages = root/'tools/ubuntu'; packages.mkdir(parents=True, exist_ok=True)
  subprocess.run(['apt-get', 'download', *PACKAGES], cwd=packages, check=True)
  private = root/'tools/ubuntu-root'; private.mkdir(exist_ok=True)
  receipts = {}
  for package in sorted(packages.glob('*.deb')):
    subprocess.run(['dpkg-deb', '-x', str(package), str(private)], check=True)
    receipts[package.name] = digest(package)
  for name in ('lib', 'lib64'):
    link = private/name
    if not link.exists(): link.symlink_to('usr/'+name, target_is_directory=True)
  binary = root/'tools/bin'; binary.mkdir(exist_ok=True)
  for name, executable in (('gcc', 'gcc'), ('g++', 'g++')):
    compiler = private/('usr/bin/x86_64-linux-gnu-'+executable+'-15')
    wrapper = binary/name
    wrapper.write_text('#!/bin/sh\nexport LD_LIBRARY_PATH='+shlex.quote(str(private/'usr/lib/x86_64-linux-gnu'))+
      '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\nexec '+shlex.quote(str(compiler))+' --sysroot='+shlex.quote(str(private))+' "$@"\n')
    wrapper.chmod(0o755)
  for name, target in (('cc', binary/'gcc'), ('c++', binary/'g++'), ('make', private/'usr/bin/make'),
                       ('rg', private/'usr/bin/rg'), ('ar', private/'usr/bin/x86_64-linux-gnu-ar')):
    link = binary/name
    if not link.exists(): link.symlink_to(target)
  # Use the host sysroot and GCC runtime for linked tests as well as CPU JIT.
  clang = root/'tools/android-sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/linux-x86_64/bin/clang'
  if clang.exists():
    wrapper = binary/'clang'
    if wrapper.is_symlink(): wrapper.unlink()
    wrapper.write_text('#!/bin/sh\nexec '+shlex.quote(str(clang))+
      ' --sysroot='+shlex.quote(str(private))+' --gcc-toolchain='+shlex.quote(str(private/'usr'))+' "$@"\n')
    wrapper.chmod(0o755)
  probe = root/'tools/compiler-probe'
  subprocess.run([str(binary/'g++'), '-x', 'c++', '-', '-o', str(probe)],
    input='#include <iostream>\nint main(){std::cout << "native compiler pass\\n";}\n', text=True, check=True)
  result = subprocess.check_output([str(probe)], text=True)
  (root/'native-toolchain.json').write_text(json.dumps({'packages': receipts, 'compile_and_execute': result,
    'qualified': False, 'vehicle_tx_enabled': False}, indent=2)+'\n')
  (root/'activate-native.sh').write_text('export PATH='+shlex.quote(str(binary))+':"$PATH"\n')


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True, type=Path)
  install(p.parse_args().root.resolve())
