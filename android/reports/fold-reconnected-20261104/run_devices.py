"""Sequential hash-bound candidate45zp bench validation with isolation receipts."""
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


def run(*command):
    subprocess.run(list(command), cwd=ROOT, check=True)


def install(device, name, label):
    device.stop()
    candidate = ROOT / 'android/build/candidates' / name
    run(ADB, '-s', device.serial, 'install', '-r', str(candidate / 'app-debug.apk'))
    run(ADB, '-s', device.serial, 'install', '-r', str(candidate / 'app-debug-androidTest.apk'))
    run(PYTHON, 'android/reports/full-model-controls-20261101/verify_installed.py', '--candidate', str(candidate),
        '--serial', device.serial, '--output', str(HERE / (label + '-installed.json')))


def experiment(device, label, kind):
    report = HERE / (label + '-isolation.json')
    if report.exists():
        raise FileExistsError(report)
    command = [PYTHON, 'android/scripts/experiment.py', '--serial', device.serial, '--adb', ADB,
               '--package', PACKAGE, '--report', str(report)]
    if kind == 'instrumentation':
        command += ['--', PYTHON, 'android/scripts/validate_instrumentation.py', '--expected-tests', '28',
                    '--output', str(HERE / (label + '.txt')), '--serial', device.serial, '--adb', ADB,
                    '--report', str(HERE / (label + '.json')), '--package', PACKAGE]
    else:
        command += ['--runtime', 'full-chain', '--', PYTHON, '-m', 'openpilot.tools.sim.android_chain_validation',
                    '--seconds', '30', '--device-replay', '--serial', device.serial, '--adb', ADB,
                    '--report', str(HERE / (label + '.json')), '--package', PACKAGE]
        if kind == 'steady':
            command.append('--steady-inputs')
    print('Starting ' + label, flush=True)
    with (HERE / (label + '.log')).open('w') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    receipt = json.loads(report.read_text())
    if receipt['isolation_status'] != 'passed':
        raise RuntimeError('Isolation failed: ' + label)
    print(json.dumps({'experiment': label, 'returncode': result.returncode, 'isolation': receipt['isolation_status']}), flush=True)


def clean(device, label):
    record = {'state': device.stop(), 'baseline': device.cool(), 'forwards': device.forwards(),
              'apk_sha256': device.artifact(), 'vehicle_tx_enabled': False}
    (HERE / (label + '-cleanup.json')).write_text(json.dumps(record, indent=2) + '\n')


def main():
    if any(line.startswith('emulator-') for line in subprocess.check_output([ADB, 'devices'], text=True).splitlines()):
        raise RuntimeError('Existing emulator must be preserved')
    phone = Device('RFCR81Y1TBJ', ADB, PACKAGE)
    try:
        install(phone, 'candidate45zo', 'phone-baseline45zo')
        experiment(phone, 'phone-baseline45zo-steady', 'steady')
        install(phone, 'candidate45zp', 'phone-candidate45zp')
        for kind in ('instrumentation', 'steady', 'faults'):
            experiment(phone, 'phone-candidate45zp-' + kind, kind)
    finally:
        clean(phone, 'phone-candidate45zp-final')
    emulator = Device('emulator-5560', ADB, PACKAGE)
    with (HERE / 'emulator.log').open('w') as log:
        process = subprocess.Popen([str(SDK / 'emulator/emulator'), '-avd', 'sunnypilot_fold3_candidate45_api35',
            '-port', '5560', '-no-window', '-no-audio', '-no-boot-anim', '-no-snapshot', '-gpu', 'host',
            '-feature', '-Vulkan', '-memory', '4096', '-cores', '4', '-camera-back', 'emulated', '-camera-front', 'emulated'],
            stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Emulator exited')
                try:
                    if emulator.command('shell', 'getprop', 'sys.boot_completed').strip() == '1':
                        break
                except subprocess.CalledProcessError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Emulator boot')
                time.sleep(2)
            install(emulator, 'candidate45zp', 'emulator-candidate45zp')
            for kind in ('instrumentation', 'steady', 'faults'):
                experiment(emulator, 'emulator-candidate45zp-' + kind, kind)
        finally:
            try:
                clean(emulator, 'emulator-candidate45zp-final')
            finally:
                try:
                    emulator.command('emu', 'kill')
                finally:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        process.wait(timeout=10)


if __name__ == '__main__':
    main()
