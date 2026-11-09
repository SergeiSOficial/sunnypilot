#!/usr/bin/env python3
"""Check a standalone 256-byte normalization graph on the Android GPU only."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from convert_snpe_linux import digest
from experiment import Device


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('adb', 'serial'):
    p.add_argument('--' + name, required=True)
  for name in ('probe', 'model', 'libraries', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
  args = p.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  base = [args.adb, '-s', args.serial]
  remote = '/data/local/tmp/fold3-normalization-probe'
  device = Device(args.serial, args.adb)
  report = {'diagnostic_only': True, 'qualified': False, 'vehicle_tx_enabled': False,
            'scope': 'single 256-value normalization graph, not road inference',
            'model_sha256': digest(args.model), 'files': {}, 'status': 'error'}
  process = None
  def adb(*command, check=True):
    return subprocess.run(base + list(command), check=check, capture_output=True, text=True, timeout=60)
  def push(path, name):
    target = remote + '/' + name
    adb('push', str(path), target)
    if adb('shell', 'sha256sum', target).stdout.split()[0] != digest(path):
      raise ValueError('Upload hash mismatch: ' + name)
    return target
  try:
    adb('shell', 'mkdir', '-p', remote)
    for path in (args.probe, *sorted(args.libraries.glob('*.so'))):
      push(path, path.name)
      report['files'][path.name] = digest(path)
    model = push(args.model, 'model.tflite')
    input_path = args.output / 'input.raw'
    values = np.arange(256, dtype=np.float32)
    values.tofile(input_path)
    push(input_path, 'input.raw')
    executable = remote + '/' + args.probe.name
    adb('shell', 'chmod', '700', executable)
    command = base + ['shell', 'timeout', '120', 'env', 'LD_LIBRARY_PATH=' + remote,
                      'FOLD3_LITERT_PRECISION=fp32', executable, model,
                      remote + '/input.raw', remote + '/output.raw']
    report['command'] = command
    with (args.output / 'run.log').open('w') as log:
      process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
      deadline = time.monotonic() + 130
      while process.poll() is None:
        device.assert_thermal_safe()
        if time.monotonic() >= deadline:
          raise TimeoutError('GPU normalization probe timed out')
        time.sleep(2)
    report['returncode'] = process.returncode
    for line in (args.output / 'run.log').read_text().splitlines():
      if line.startswith('{'):
        report['runtime'] = json.loads(line)
    if process.returncode == 0 and report.get('runtime', {}).get('gpu_only') is True:
      result_path = args.output / 'output.raw'
      adb('pull', remote + '/output.raw', str(result_path))
      actual = np.fromfile(result_path, np.float32)
      expected = (values - np.float32(127.5)) / np.float32(63.75)
      if actual.shape != expected.shape:
        raise ValueError('Output size mismatch')
      report.update(output_sha256=digest(result_path), nonfinite=int(np.count_nonzero(~np.isfinite(actual))),
                    unequal=int(np.count_nonzero(actual != expected)),
                    tolerance_mismatches=int(np.count_nonzero(~np.isclose(actual, expected, rtol=1e-3, atol=1e-4) | ~np.isfinite(actual))),
                    max_abs_error=float(np.max(np.abs(actual - expected))))
      report['status'] = 'pass' if report['unequal'] == 0 and report['nonfinite'] == 0 else 'fail'
    else:
      report['status'] = 'execution_failed_or_cpu_partition'
  except Exception as exc:
    report['error'] = f'{type(exc).__name__}: {exc}'
  finally:
    if process is not None and process.poll() is None:
      listing = adb('shell', 'ps', '-A', '-o', 'PID,ARGS', check=False).stdout
      for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].isdigit() and fields[1] == executable:
          adb('shell', 'kill', fields[0], check=False)
      process.terminate()
      try:
        process.wait(timeout=10)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    adb('shell', 'rm', '-rf', remote, check=False)
    report['remote_directory_removed'] = adb('shell', 'test', '!', '-e', remote, check=False).returncode == 0
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))
  return 0 if report['status'] == 'pass' and report['remote_directory_removed'] else 1


if __name__ == '__main__':
  raise SystemExit(main())
