"""Bind diagnostic failures and final candidate recovery evidence without promotion."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'android/reports/full-model-controls-20261101'))
from build_summary import controls


def read(path):
  return json.loads(Path(path).read_text())


def inventory(path, apk):
  rows = read(path)
  for row in rows:
    assert row['status'] != 'pending', path
    cmd = row['command']
    isolation = read(cmd[cmd.index('--report') + 1])
    assert isolation['apk_sha256'] == apk and isolation['artifact_stable'], row['name']
    assert isolation['isolation_status'] == 'passed', row['name']
  return {'status': 'passed' if all(r['status'] == 'passed' for r in rows) else 'failed',
          'checks': [{k: r[k] for k in ('name', 'status', 'isolation_status', 'log')} for r in rows]}


def main():
  candidate = 'candidate45zg'
  artifact = read(ROOT / 'android/build/candidates' / candidate / 'artifacts.json')
  apk = artifact['files']['app-debug.apk']['sha256']
  devices = {}
  for device in ('fold', 'emulator'):
    installed = read(HERE / f'{candidate}-installed-{device}.json')
    assert installed['status'] == 'passed'
    assert installed['files']['app-debug.apk']['sha256'] == apk
    devices[device] = inventory(HERE / f'{device}-{candidate}-validation-driver.json', apk)
    devices[device]['installed_hash_status'] = 'passed'
  acceleration = inventory(HERE / f'fold-{candidate}-accuracy-validation-driver.json', apk)
  mnn = read(HERE / f'fold-{candidate}-mnn-native-initial.json')['models']['dmonitoring_model']
  assert mnn['identity']['sha256']['apk'] == apk
  assert mnn['assignment']['no_cpu_fallback_verified'] is True
  mnn_checks = [mnn['parity'], *mnn['additional_seed_parity'].values()]
  mnn['initial_seed_checks'] = mnn_checks
  ncnn = read(HERE / f'fold-{candidate}-ncnn-vulkan-models.json')['models']
  ncnn_driver = ncnn['dmonitoring_model']
  assert ncnn_driver['identity']['sha256']['apk'] == apk
  assert ncnn_driver['assignment']['no_cpu_fallback_verified'] is True
  cleanup = read(HERE / f'{candidate}-final-cleanup.json')
  assert 'emulator_cleanup_error' not in cleanup
  for device in ('fold', 'emulator'):
    record = cleanup[device]
    assert not record['state']['processes'] and not record['state']['camera_owned']
    assert not record['forwards'] and record['baseline']['normal_thermal_seconds'] >= 30
    assert record['baseline']['thermal_status'] == 0 and record['baseline']['skin_c'] <= 32
  vulkan = []
  for label in ('trace', 'rne', 'rne-deep', 'cube', 'head', 'reduction'):
    source = HERE / f'ncnn-driver-seed42-{label}.json'
    report = read(source)
    vulkan.append({'evidence': source.name, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
      'status': report['status'], 'installed_apk_sha256': report['apk_sha256'],
      'provider_sha256': report['sha256']['libncnn.so'], 'parity': report.get('parity'),
      'diagnostic_only': True, 'qualified': False})
  comparison = []
  for name, source in [('candidate44', ROOT / 'android/reports/npu-recovery-20261101/fold-candidate44-chain-device-steady.json'),
                       ('candidate45ze', ROOT / 'android/reports/performance-followup-20261103/fold-candidate45ze-chain-device-steady.json'),
                       (candidate, HERE / f'fold-{candidate}-chain-device-steady.json')]:
    comparison.append(controls(name, source))
  assert len({r['fixture_sha256'] for r in comparison}) == 1
  result = {'candidate': candidate, 'apk_sha256': apk, 'branch': 'codex/fold3-android-port',
    'vehicle_tx_enabled': False, 'physical_engagement_allowed': False, 'qualification_status': 'blocked',
    'host_tests': {'status': 'passed', 'count': 186, 'evidence': 'host-tests-final2.log'},
    'android_build': {'status': 'passed', 'log': '../../build/candidate45zg-build.log'},
    'devices': devices, 'accelerator_validation': acceleration,
    'mnn_native_driver': {'status': 'failed', 'comparisons_run': 3, 'failed': 3,
       'required_driver_comparisons': 159, 'total_original_reference_comparisons_required': 795,
       'rtol': 1e-3, 'atol': 1e-4, 'identity': mnn['identity'], 'seed_checks': mnn_checks,
       'remaining_sequence_replay_and_timing': 'blocked', 'no_cpu_fallback': True,
       'same_input_gemv': read(HERE / 'mnn-native-same-input-gemv.json')},
    'ncnn_vulkan_diagnostics': vulkan, 'matched_controls': comparison,
    'ncnn_final_apk': {'driver': {'status': ncnn_driver['status'], 'identity': ncnn_driver['identity'],
       'seed_checks': [ncnn_driver['parity'], *ncnn_driver['additional_seed_parity'].values()],
       'no_cpu_fallback': True, 'qualified': False},
       'road': {'status': ncnn['driving_supercombo']['status'],
                'error': ncnn['driving_supercombo'].get('error'),
                'reason': 'No converted ncnn road entry in this APK'}},
    'cleanup': cleanup,
    'combined_workloads': {'status': 'blocked', 'stages': ['30 seconds', '2 minutes', '30 minutes'],
       'reason': 'Full-model accuracy, inference timing, camera freshness and complete-cycle controls gates remain uncleared'},
    'retained_baselines': ['candidate35', 'candidate40', 'candidate44', 'candidate45ze', 'candidate45zf'],
    'historical_accuracy': {'candidate': 'candidate45ze', 'comparisons': 795, 'status': 'passed',
       'evidence': '../performance-followup-20261103/candidate45ze-accuracy.json', 'qualifies_new_apk': False},
    'physical_gates': {'rear_sensor_identity': 'blocked', 'mounted_calibration': 'blocked', 'Panda_vehicle': 'blocked'},
    'cooling_optimization': 'deferred', 'thermal_safeguards': 'enforced',
    'notes': ['Functional fixture success does not qualify camera perception, accelerator timing or vehicle use',
              'Vulkan shell probes bind their provider hash separately from the installed APK',
              'Candidate45zf lacks the portable reduction patch; candidate45zg packages it',
              'MNN native convolution remains diagnostic; strict accurate reference preserved in candidate45ze']}
  (HERE / f'{candidate}-validation.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps({'candidate': candidate, 'devices': {k:v['status'] for k,v in devices.items()},
                    'qualification_status': 'blocked'}))


if __name__ == '__main__':
  main()
