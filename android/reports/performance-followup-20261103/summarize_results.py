#!/usr/bin/env python3
"""Bind the follow-up evidence without promoting partial timing or fixtures."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'android/reports/full-model-controls-20261101'))
from build_latest_summary import accelerator_timing
from build_summary import controls


def read(path):
  return json.loads(path.read_text())


def main():
  artifact = read(ROOT / 'android/build/candidates/candidate45ze/artifacts.json')
  apk = artifact['files']['app-debug.apk']['sha256']
  accuracy_path = HERE / 'candidate45ze-accuracy.json'
  accuracy = read(accuracy_path)
  assert accuracy['apk_sha256'] == apk and accuracy['status'] == 'passed'
  assert accuracy['comparisons_run'] == 795 and accuracy['failed'] == 0
  timings = {}
  for model in ('driver', 'road'):
    timings.update(accelerator_timing(HERE / f'fold45ze-mnn-{model}-timing.json', apk, accuracy['identities']))
  devices = {}
  for device in ('fold', 'emulator'):
    installed_path = HERE / (f'candidate45ze-installed-{device}-retry1.json' if device == 'emulator' else f'candidate45ze-installed-{device}.json')
    installed = read(installed_path)
    assert installed['status'] == 'passed'
    assert installed['files']['app-debug.apk']['sha256'] == apk
    inventory_path = HERE / (f'{device}-candidate45ze-retry1-validation-driver.json' if device == 'emulator'
                             else f'{device}-candidate45ze-validation-driver.json')
    inventory = read(inventory_path)
    checks = []
    for row in inventory:
      assert row['status'] != 'pending', 'Do not publish unfinished inventories'
      command = row['command']
      isolation_path = Path(command[command.index('--report') + 1])
      isolation = read(isolation_path)
      assert isolation['apk_sha256'] == apk and isolation['artifact_stable']
      assert isolation['isolation_status'] == 'passed'
      checks.append({'name': row['name'], 'status': row['status'], 'isolation_status': isolation['isolation_status'],
                     'isolation': isolation_path.name})
    devices[device] = {'functional_status': 'passed' if all(c['status'] == 'passed' for c in checks) else 'failed',
                       'checks': checks, 'inventory': inventory_path.name, 'installed_hash_status': 'passed',
                       'installed_evidence': installed_path.name,
                       'retained_attempts': ['emulator-candidate45ze-validation-driver.json',
                           'candidate45ze-emulator-attempt1-interrupted-cleanup.json'] if device == 'emulator' else []}
  comparisons = []
  for candidate, path in (
      ('candidate44', ROOT / 'android/reports/npu-recovery-20261101/fold-candidate44-chain-device-steady.json'),
      ('candidate45zd', ROOT / 'android/reports/full-model-controls-20261101/fold-candidate45zd-chain-device-steady.json'),
      ('candidate45ze', HERE / 'fold-candidate45ze-chain-device-steady.json')):
    comparisons.append(controls(candidate, path))
  assert len({r['fixture_sha256'] for r in comparisons}) == 1
  summary = {'candidate': 'candidate45ze', 'apk_sha256': apk, 'branch': 'codex/fold3-android-port',
      'vehicle_tx_enabled': False, 'physical_engagement_allowed': False, 'qualification_status': 'blocked',
      'accuracy': {'status': 'passed', 'comparisons': 795, 'no_cpu_fallback': True, 'rtol': 1e-3, 'atol': 1e-4,
                   'evidence': accuracy_path.name, 'sha256': hashlib.sha256(accuracy_path.read_bytes()).hexdigest()},
      'accelerator_timing': timings, 'matched_controls': comparisons, 'devices': devices,
      'combined_visual_workloads': {'status': 'blocked', 'stages': ['30 seconds', '2 minutes', '30 minutes'],
          'reason': 'Inference timing, complete-cycle controls timing and physical camera gates are not cleared'},
      'physical_gates': {'rear_sensor_identity': 'blocked', 'mounted_calibration': 'blocked',
                        'Panda_vehicle_validation': 'blocked'},
      'cooling_optimization': 'deferred', 'thermal_safeguards': 'enforced',
      'retained_baselines': ['candidate35', 'candidate40', 'candidate44', 'candidate45zd'],
      'notes': ['Functional fixture success cannot qualify perception or physical vehicle operation',
                'Blocked road timing has no completed p99; last-call wall stages are diagnostic only',
                'Matched fixture does not establish causality across differing CPU policies or device conditions']}
  (HERE / 'candidate45ze-validation.json').write_text(json.dumps(summary, indent=2) + '\n')
  print(json.dumps({'accuracy': 'passed', 'devices': {k:v['functional_status'] for k,v in devices.items()},
                    'qualification_status': 'blocked'}))


if __name__ == '__main__':
  main()
