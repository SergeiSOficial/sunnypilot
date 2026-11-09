#!/usr/bin/env python3
"""Run only inside experiment.py; shell-domain SNPE capability, not app qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--adb', required=True)
  parser.add_argument('--serial', required=True)
  parser.add_argument('--probe', required=True, type=Path)
  parser.add_argument('--library', required=True, type=Path)
  parser.add_argument('--output', required=True, type=Path)
  args = parser.parse_args()
  remote = '/data/local/tmp/fold3-c74-snpe-probe'
  def adb(*command):
    return subprocess.run([args.adb, '-s', args.serial, *command], check=True, text=True,
                          capture_output=True, timeout=60).stdout
  report = {'execution_domain': 'adb shell, not Android app', 'qualified': False,
            'vehicle_tx_enabled': False, 'files': {}}
  try:
    adb('shell', 'mkdir', '-p', remote)
    for source, name in ((args.probe, 'probe'), (args.library, 'libSNPE.so')):
      expected = hashlib.sha256(source.read_bytes()).hexdigest()
      adb('push', str(source), remote+'/'+name)
      actual = adb('shell', 'sha256sum', remote+'/'+name).split()[0]
      if expected != actual: raise ValueError('Device probe artifact mismatch')
      report['files'][name] = expected
    adb('shell', 'chmod', '700', remote+'/probe')
    result = subprocess.run([args.adb, '-s', args.serial, 'shell', 'env',
      'LD_LIBRARY_PATH='+remote, remote+'/probe', remote+'/libSNPE.so'],
      text=True, capture_output=True, timeout=60)
    report.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode == 0: report['capability'] = json.loads(result.stdout)
  finally:
    adb('shell', 'rm', '-rf', remote)
    # test returns nonzero when the exact owned directory has been removed.
    check = subprocess.run([args.adb, '-s', args.serial, 'shell', 'test', '!', '-e', remote], timeout=30)
    report['remote_directory_removed'] = check.returncode == 0
    args.output.write_text(json.dumps(report, indent=2)+'\n')
  return 0 if report.get('returncode') == 0 and report['remote_directory_removed'] else 1


if __name__ == '__main__': raise SystemExit(main())
