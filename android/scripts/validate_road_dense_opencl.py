#!/usr/bin/env python3
"""Run an isolated GPU-only captured dense probe; not model qualification."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from experiment import Device
from prepare_road_dense_opencl_probe import sha


def fixture_paths(fixtures):
  manifest = json.loads((fixtures / 'manifest.json').read_text())
  if set(manifest['files']) != {'input', 'weights', 'bias', 'expected'}:
    raise ValueError('Unexpected fixture file set')
  paths = []
  for name, item in manifest['files'].items():
    if item['file'] != name + '.f32':
      raise ValueError('Unexpected fixture filename: ' + name)
    path = fixtures / item['file']
    if sha(path) != item['sha256']:
      raise ValueError('Local fixture hash mismatch: ' + path.name)
    paths.append(path)
  return paths


def run(args):
  args.output.mkdir(parents=True, exist_ok=False)
  base = [args.adb, '-s', args.serial]
  remote = '/data/local/tmp/fold3-road-dense-probe'
  device = Device(args.serial, args.adb)
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'gpu_only': True,
            'fixture_manifest_sha256': sha(args.fixtures / 'manifest.json'), 'files': {}, 'status': 'error'}
  process = None

  def adb(*parts, check=True):
    return subprocess.run(base + list(parts), capture_output=True, text=True, check=check, timeout=120)

  try:
    paths = [args.probe, args.kernel, *fixture_paths(args.fixtures)]
    adb('shell', 'mkdir', '-p', remote)
    for path in paths:
      adb('push', str(path), remote + '/' + path.name)
      digest = sha(path)
      if adb('shell', 'sha256sum', remote + '/' + path.name).stdout.split()[0] != digest:
        raise ValueError('Device file hash mismatch: ' + path.name)
      report['files'][path.name] = digest
    executable = remote + '/' + args.probe.name
    adb('shell', 'chmod', '700', executable)
    command = base + ['shell', 'timeout', '60', executable, remote, remote + '/' + args.kernel.name]
    if args.parallel:
      command.append('--parallel')
    if args.save_output:
      command.append('--save-output')
    report['command'] = command
    with (args.output / 'run.log').open('w') as log:
      process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
      deadline = time.monotonic() + 70
      while process.poll() is None:
        device.assert_thermal_safe()
        if time.monotonic() >= deadline:
          raise TimeoutError('Dense probe timed out')
        time.sleep(1)
      report['returncode'] = process.returncode
    for line in (args.output / 'run.log').read_text().splitlines():
      if line.startswith('{'):
        report['result'] = json.loads(line)
    result = report.get('result', {})
    report['status'] = ('pass' if report['returncode'] == 0 and result.get('gpu_only') is True
                        and result.get('mismatches') == 0 and result.get('nonfinite') == 0 else 'fail')
    if args.save_output and report['status'] == 'pass':
      output_path = args.output / 'actual.f32'
      adb('pull', remote + '/actual.f32', str(output_path))
      if output_path.stat().st_size != 2048 * 4:
        raise ValueError('Unexpected device output size')
      report['device_output_sha256'] = sha(output_path)
  except Exception as exc:
    report['status'] = 'error'
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
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('adb', 'serial'):
    p.add_argument('--' + name, required=True)
  for name in ('probe', 'kernel', 'fixtures', 'output'):
    p.add_argument('--' + name, type=Path, required=True)
  p.add_argument('--parallel', action='store_true')
  p.add_argument('--save-output', action='store_true')
  raise SystemExit(run(p.parse_args()))
