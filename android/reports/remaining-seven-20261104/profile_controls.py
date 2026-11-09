"""Scheduling diagnostic only: system tracing adds overhead; never qualification."""
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ADB = str(Path.home() / 'Library/Android/sdk/platform-tools/adb')
SERIAL = 'RFCR81Y1TBJ'
PACKAGE = 'com.samsung.sunnypilot.fold3'


def main():
    command = [ADB, '-s', SERIAL]
    if subprocess.check_output(command + ['shell', 'cat', '/sys/kernel/tracing/tracing_on'], text=True).strip() != '0':
        raise RuntimeError('Preserve existing system tracing session')
    trace = HERE / 'controls-baseline45zp-sched.txt'
    if trace.exists():
        raise FileExistsError(trace)
    subprocess.run(command + ['shell', 'atrace', '--async_start', '-b', '32768', 'sched'], check=True)
    try:
        result = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'openpilot.tools.sim.android_chain_validation',
            '--seconds', '30', '--device-replay', '--steady-inputs', '--serial', SERIAL, '--adb', ADB,
            '--package', PACKAGE, '--report', str(HERE / 'controls-baseline45zp-profile.json')], cwd=ROOT)
        report = json.loads((HERE / 'controls-baseline45zp-profile.json').read_text())
        pid = report['runtime_reports']['controlsd']['pid']
        with (HERE / 'controls-baseline45zp-raw.jsonl').open('wb') as output:
            subprocess.run(command + ['exec-out', 'run-as', PACKAGE, 'cat', f'files/controlsd-timing-{pid}.jsonl'],
                           stdout=output, check=True)
        return result.returncode
    finally:
        with trace.open('wb') as output:
            subprocess.run(command + ['shell', 'atrace', '--async_stop', 'sched'], stdout=output, check=True)
        if subprocess.check_output(command + ['shell', 'cat', '/sys/kernel/tracing/tracing_on'], text=True).strip() != '0':
            raise RuntimeError('System tracing did not stop')


if __name__ == '__main__':
    sys.exit(main())
