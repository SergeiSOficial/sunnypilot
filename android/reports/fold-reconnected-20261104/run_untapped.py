"""Trace-free seed42 diagnostics, sequentially, with explicit emulator ownership."""
import json
import subprocess
import time

from run_devices import ADB, HERE, PYTHON, ROOT, SDK, Device, clean, run


def probe(device, label):
    report = HERE / (label + '.json')
    if report.exists():
        raise FileExistsError(report)
    with (HERE / (label + '.log')).open('w') as log:
        result = subprocess.run([PYTHON, 'android/scripts/probe_ncnn.py', '--serial', device.serial,
            '--adb', ADB, '--seed', '42', '--trace-count', '0', '--report', str(report)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    receipt = json.loads(report.read_text())
    if receipt.get('cleanup_error') or not receipt.get('artifact_stable'):
        raise RuntimeError('Probe cleanup/artifact failure: ' + label)
    print(json.dumps({'experiment': label, 'returncode': result.returncode,
                      'status': receipt['status'], 'parity': receipt.get('parity')}), flush=True)


def main():
    if any(line.startswith('emulator-') for line in subprocess.check_output([ADB, 'devices'], text=True).splitlines()):
        raise RuntimeError('Existing emulator must be preserved')
    probe(Device('RFCR81Y1TBJ', ADB, 'com.samsung.sunnypilot.fold3'), 'phone-untapped45zp')
    emulator = Device('emulator-5560', ADB, 'com.samsung.sunnypilot.fold3')
    with (HERE / 'emulator-software-vulkan.log').open('w') as log:
        process = subprocess.Popen([str(SDK / 'emulator/emulator'), '-avd', 'sunnypilot_fold3_candidate45_api35',
            '-port', '5560', '-no-window', '-no-audio', '-no-boot-anim', '-no-snapshot', '-gpu', 'swiftshader',
            '-feature', 'Vulkan', '-memory', '4096', '-cores', '4', '-camera-back', 'emulated', '-camera-front', 'emulated'],
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
            run(PYTHON, 'android/reports/full-model-controls-20261101/verify_installed.py', '--candidate',
                str(ROOT / 'android/build/candidates/candidate45zp'), '--serial', emulator.serial,
                '--output', str(HERE / 'emulator-untapped45zp-installed.json'))
            probe(emulator, 'emulator-untapped45zp')
        finally:
            try:
                clean(emulator, 'emulator-untapped45zp-final')
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
