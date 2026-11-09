#!/usr/bin/env python3
"""Diagnostic-only full-graph GPU compilation; does not qualify inference."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from convert_snpe_linux import digest
from experiment import Device


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('adb', 'serial'): p.add_argument('--' + name, required=True)
  for name in ('probe', 'model', 'libraries', 'output'): p.add_argument('--' + name, type=Path, required=True)
  p.add_argument('--precision', choices=('fp32', 'fp16-fp32-accum'), required=True)
  args = p.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  remote = '/data/local/tmp/fold3-litert-full-compile'
  base = [args.adb, '-s', args.serial]
  device = Device(args.serial, args.adb)
  def adb(*parts, check=True):
    return subprocess.run(base + list(parts), capture_output=True, text=True, check=check, timeout=180)
  def push(source, name):
    target = remote + '/' + name
    adb('push', str(source), target)
    if adb('shell', 'sha256sum', target).stdout.split()[0] != digest(source):
      raise ValueError('Uploaded artifact hash mismatch: ' + name)
    return target
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'execution_domain': 'ADB shell, not app',
            'test': 'full model compilation only; no inference or parity', 'precision': args.precision,
            'model_sha256': digest(args.model), 'files': {}, 'status': 'error'}
  process = None
  try:
    adb('shell', 'mkdir', '-p', remote)
    for source in (args.probe, *sorted(args.libraries.glob('*.so'))):
      push(source, source.name); report['files'][source.name] = digest(source)
    model = push(args.model, 'model.tflite')
    executable = remote + '/' + args.probe.name
    adb('shell', 'chmod', '700', executable)
    command = base + ['shell', 'timeout', '120', 'env', 'LD_LIBRARY_PATH=' + remote,
                      'FOLD3_LITERT_PRECISION=' + args.precision, executable, model, '--compile-only']
    report['command'] = command
    with (args.output / 'run.log').open('w') as log:
      process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
      deadline = time.monotonic() + 130
      while process.poll() is None:
        device.assert_thermal_safe()
        if time.monotonic() >= deadline: raise TimeoutError('Full graph compile timed out')
        time.sleep(2)
      report['returncode'] = process.returncode
    lines = (args.output / 'run.log').read_text().splitlines()
    for line in lines:
      if line.startswith('{'): report['runtime'] = json.loads(line)
    report['status'] = ('compile_pass' if report['returncode'] == 0 and report.get('runtime', {}).get('gpu_only') is True
                        else 'compile_failed_or_cpu_partition')
  except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
  finally:
    if process is not None and process.poll() is None:
      # Kill only processes whose argv[0] is this diagnostic executable.
      listing = adb('shell', 'ps', '-A', '-o', 'PID,ARGS', check=False).stdout
      for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit() and fields[1] == executable:
          adb('shell', 'kill', fields[0], check=False)
      process.terminate()
      try: process.wait(timeout=10)
      except subprocess.TimeoutExpired: process.kill(); process.wait()
    adb('shell', 'rm', '-rf', remote, check=False)
    report['remote_directory_removed'] = adb('shell', 'test', '!', '-e', remote, check=False).returncode == 0
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))
  return 0 if report['status'] == 'compile_pass' and report['remote_directory_removed'] else 1


if __name__ == '__main__': raise SystemExit(main())
