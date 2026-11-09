#!/usr/bin/env python3
"""Extract pinned QAIRT converter components into a user-owned Linux workspace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile

VERSION = '2.47.0.260601'
SDK_SHA256 = 'd3497e110eae82c35a9152a93c0a18bbede402aaf9faa7a97c8079eb0f522b01'


def environment(root):
  sdk = root/'sdk/qairt'/VERSION
  python_lib = (root/'converter-venv/bin/python').resolve().parent.parent/'lib'
  env = dict(os.environ)
  env.update(SNPE_ROOT=str(sdk), QAIRT_SDK_ROOT=str(sdk),
             PYTHONPATH=str(sdk/'lib/python'),
             LD_LIBRARY_PATH=os.pathsep.join(map(str, (sdk/'lib/x86_64-linux-clang', python_lib,
                                                       root/'sysroot/usr/lib/x86_64-linux-gnu'))),
             PATH=str(root/'converter-venv/bin')+os.pathsep+env['PATH'])
  return sdk, env


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--root', required=True, type=Path)
  parser.add_argument('--check-only', action='store_true', help='Check an already extracted SDK')
  args = parser.parse_args(); root = args.root.resolve()
  if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('SNPE host requires Linux x86_64')
  archive = root/'downloads'/f'v{VERSION}.zip'
  checksum_hash = hashlib.sha256()
  with archive.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''): checksum_hash.update(chunk)
  checksum = checksum_hash.hexdigest()
  if checksum != SDK_SHA256: raise ValueError('Pinned SDK archive hash mismatch')
  prefix = f'qairt/{VERSION}/'
  with zipfile.ZipFile(archive) as source:
    for member in source.infolist():
      if args.check_only: break
      if member.filename == 'qairt/' and member.is_dir(): continue
      if not member.filename.startswith(prefix): raise ValueError('Unexpected SDK archive root')
      relative = Path(member.filename[len(prefix):])
      if '..' in relative.parts or relative.is_absolute(): raise ValueError('Unsafe SDK member')
      if not str(relative).startswith(('lib/python/', 'lib/x86_64-linux-clang/', 'bin/x86_64-linux-clang/')):
        continue
      target = root/'sdk'/member.filename
      if member.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
      target.parent.mkdir(parents=True, exist_ok=True)
      target.write_bytes(source.read(member))
      target.chmod(0o755 if str(relative).startswith('bin/') else 0o644)
  sdk, env = environment(root)
  tool = sdk/'bin/x86_64-linux-clang/snpe-onnx-to-dlc'
  result = subprocess.run([str(root/'converter-venv/bin/python'), str(tool), '--help'], env=env,
                          capture_output=True, text=True, timeout=120)
  (root/'converter-help.log').write_text(result.stdout+result.stderr)
  # Ubuntu 26.04 is deliberately identified; do not spoof the SDK's supported
  # Ubuntu 22.04/24.04 dependency-check environment.
  report = {'qairt_version': VERSION, 'sdk_archive_sha256': checksum,
            'os_release': Path('/etc/os-release').read_text(),
            'vendor_supported_host': 'VERSION_ID="22.04"' in Path('/etc/os-release').read_text() or
                                     'VERSION_ID="24.04"' in Path('/etc/os-release').read_text(),
            'converter_help_returncode': result.returncode, 'qualified': False, 'vehicle_tx_enabled': False}
  (root/'converter-host.json').write_text(json.dumps(report, indent=2)+'\n')
  print(json.dumps(report), flush=True)
  return result.returncode


if __name__ == '__main__': raise SystemExit(main())
