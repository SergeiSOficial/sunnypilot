#!/usr/bin/env python3
"""Bind the installed application and instrumentation to an archived candidate."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--serial', required=True)
parser.add_argument('--adb', default='/Users/sergios/Library/Android/sdk/platform-tools/adb')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--candidate', type=Path, default=ROOT / 'android/build/candidates/candidate45')
args = parser.parse_args()
adb = [args.adb, '-s', args.serial]
manifest = json.loads((args.candidate / 'artifacts.json').read_text())
result = {'serial': args.serial, 'candidate': manifest['candidate'], 'vehicle_tx_enabled': False, 'files': {}}
for package, filename in [('com.samsung.sunnypilot.fold3', 'app-debug.apk'),
                          ('com.samsung.sunnypilot.fold3.test', 'app-debug-androidTest.apk')]:
    paths = subprocess.check_output([*adb, 'shell', 'pm', 'path', package], text=True).splitlines()
    base = next(line.removeprefix('package:').strip() for line in paths if line.endswith('/base.apk'))
    digest = subprocess.check_output([*adb, 'shell', 'sha256sum', base], text=True).split()[0]
    expected = manifest['files'][filename]['sha256']
    result['files'][filename] = {'package': package, 'installed_path': base, 'sha256': digest,
                                  'expected_sha256': expected, 'status': 'passed' if digest == expected else 'failed'}
result['fingerprint'] = subprocess.check_output([*adb, 'shell', 'getprop', 'ro.build.fingerprint'], text=True).strip()
result['status'] = 'passed' if all(item['status'] == 'passed' for item in result['files'].values()) else 'failed'
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'serial': args.serial, 'status': result['status']}))
raise SystemExit(0 if result['status'] == 'passed' else 1)
