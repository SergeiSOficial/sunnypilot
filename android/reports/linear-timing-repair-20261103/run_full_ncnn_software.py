"""Sequential final accelerator rejection checks, emulator fixtures, and cleanup."""
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'android/scripts'))
from experiment import Device

PYTHON = str(ROOT / '.venv/bin/python')
SDK = Path.home() / 'Library/Android/sdk'
ADB = str(SDK / 'platform-tools/adb')
PACKAGE = 'com.samsung.sunnypilot.fold3'
CANDIDATE = ROOT / 'android/build/candidates/candidate45zj'


def run(*command):
  subprocess.run(list(command), cwd=ROOT, check=True)


def main():
  cleanup = {'vehicle_tx_enabled': False, 'fold_status': 'blocked: disconnected'}
  # A second emulator would invalidate the isolation assumptions.
  devices = subprocess.check_output([ADB, 'devices'], text=True)
  if any(line.startswith('emulator-') for line in devices.splitlines()):
    raise RuntimeError('An emulator is already running; preserve it and inspect before proceeding')
  emulator = Device('emulator-5560', ADB, PACKAGE)
  log = (ROOT / 'android/build/full-ncnn-software-emulator.log').open('w')
  process = subprocess.Popen([str(SDK / 'emulator/emulator'), '-avd', 'sunnypilot_fold3_candidate45_api35',
    '-port', '5560', '-no-window', '-no-audio', '-no-boot-anim', '-no-snapshot', '-gpu', 'swiftshader',
    '-feature', 'Vulkan', '-memory', '4096', '-cores', '4', '-camera-back', 'emulated', '-camera-front', 'emulated'],
    stdout=log, stderr=subprocess.STDOUT)
  try:
    deadline = time.monotonic() + 180
    while True:
      if process.poll() is not None:
        raise RuntimeError('Dedicated emulator exited before boot')
      try:
        if emulator.command('shell', 'getprop', 'sys.boot_completed').strip() == '1':
          break
      except subprocess.CalledProcessError:
        pass
      if time.monotonic() >= deadline:
        raise TimeoutError('Dedicated emulator did not boot')
      time.sleep(2)
    run(ADB, '-s', emulator.serial, 'install', '-r', str(CANDIDATE / 'app-debug.apk'))
    run(ADB, '-s', emulator.serial, 'install', '-r', str(CANDIDATE / 'app-debug-androidTest.apk'))
    run(PYTHON, 'android/reports/full-model-controls-20261101/verify_installed.py', '--candidate', str(CANDIDATE),
        '--serial', emulator.serial, '--output', str(HERE / 'full-ncnn-software-installed.json'))
    results = []
    for seed in (42, 7, 123):
      report = HERE / f'ncnn-software-seed{seed}.json'
      result = subprocess.run([PYTHON, 'android/scripts/probe_ncnn.py', '--serial', emulator.serial,
        '--adb', ADB, '--seed', str(seed), '--trace-count', '0', '--report', str(report)], cwd=ROOT)
      record = json.loads(report.read_text())
      results.append({'seed': seed, 'returncode': result.returncode, 'status': record['status'],
        'report': report.name, 'software_vulkan': True, 'qualified': False})
      (HERE / 'full-ncnn-software-results.json').write_text(json.dumps(results, indent=2) + '\n')
      if result.returncode:
        break  # Do not advance after a correctness/cleanup failure.

  finally:
    try:
      cleanup['emulator'] = {'state': emulator.stop(), 'baseline': emulator.cool(), 'forwards': emulator.forwards()}
      cleanup['emulator']['apk_sha256'] = emulator.artifact()
    except Exception as error:
      cleanup['emulator_cleanup_error'] = f'{type(error).__name__}: {error}'
    finally:
      try:
        emulator.command('emu', 'kill')
      finally:
        try:
          process.wait(timeout=30)
        except subprocess.TimeoutExpired:
          process.terminate()
          process.wait(timeout=10)
        log.close()
        cleanup['emulator_exit_code'] = process.returncode
        (HERE / 'full-ncnn-software-cleanup.json').write_text(json.dumps(cleanup, indent=2) + '\n')
  if 'emulator_cleanup_error' in cleanup:
    raise RuntimeError(cleanup['emulator_cleanup_error'])


if __name__ == '__main__':
  main()
