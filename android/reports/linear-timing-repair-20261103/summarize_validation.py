"""Summarize immutable APK-bound diagnostics without promoting blocked hardware."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def read(path):
  return json.loads(path.read_text())

inventory = read(HERE / 'emulator-candidate45zj-validation-driver.json')
comparison = []
for name in ('candidate45zg', 'candidate45zi'):
  path = HERE / (name + '-controls-ab.json')
  if not path.exists():
    continue
  fixture = read(path)
  controls = fixture['runtime_reports']['controlsd']
  installed = read(HERE / (name + '-controls-ab-installed.json'))
  comparison.append({'candidate': name, 'apk_sha256': installed['files']['app-debug.apk']['sha256'],
    'fixture_sha256': fixture['fixture_sha256'], 'fixture_status': fixture['status'],
    'source_deadline_misses': fixture['source_deadline_misses'],
    'controls': {key: controls.get(key) for key in ('p99_work_ms', 'p99_complete_cycle_ms', 'mean_cycle_hz',
       'cycle_deadline_misses', 'scheduled_deadline_misses', 'skipped_slots_total', 'android_controls_gil_switch_interval_s')},
    'p99_stages_ms': controls.get('p99_stages_ms')})
if len(comparison) == 2:
  assert comparison[0]['fixture_sha256'] == comparison[1]['fixture_sha256']
record = {'candidate': 'candidate45zj', 'apk_sha256': read(HERE/'candidate45zj-package-check.json')['apk_sha256'],
 'vehicle_tx_enabled': False, 'physical_engagement_allowed': False,
 'host_tests': {'status':'passed','count':188,'log':'host-tests-final-trace.log'},
 'conversion_reference': {'status':'passed','comparisons':159,'accelerator_qualification':False},
 'shader_compilation': read(HERE/'linear-shader-compilation.json'),
 'shader_host_arithmetic': read(HERE/'linear-shader-host-parity.json'),
 'emulator_experiments': [{key:r.get(key) for key in ('name','status','isolation_status','log')} for r in inventory],
 'installed_verification': read(HERE/'candidate45zj-installed-emulator.json')['status'],
 'controls_ab': comparison, 'controls_gil_experiment':'reverted after paired-run regression; timing unresolved',
 'fold3_accelerator': {'status':'blocked','reason':'Device disconnected; no new device inference or full 795-comparison result'},
 'vehicle_qualification': {'status':'blocked','reason':'Unmounted phone, no independent rear-sensor proof, Panda disconnected'},
 'cooling_optimization': 'deferred; thermal safeguards retained',
 'excluded_candidate': {'name':'candidate45zh',
   'reason':'Archive created before build completed; old APK with new graph/provider artifacts; zero completed tests'},
 'broad_lint': {'status':'failed','log':'lint.log','reason':'14 findings in existing portions of accelerator/converter files; focused new checks pass'}}
for name, filename in [('emulator_cleanup','candidate45zj-final-cleanup.json'),('controls_ab_cleanup','controls-ab-final-cleanup.json')]:
  path=HERE/filename
  record[name]=read(path) if path.exists() else {'status':'pending'}
probe = HERE/'linear-software-vulkan.json'
record['software_vulkan'] = read(probe) if probe.exists() else {'status':'pending'}
cleanup = HERE/'linear-vulkan-final-cleanup.json'
record['software_vulkan_cleanup'] = read(cleanup) if cleanup.exists() else {'status':'pending'}
for key, filename in [('full_model_software_vulkan', 'full-ncnn-software-results.json'),
                      ('full_model_software_cleanup', 'full-ncnn-software-cleanup.json'),
                      ('head_trace', 'ncnn-software-head-seed42-comparison.json'),
                      ('head_trace_cleanup', 'head-ncnn-software-cleanup.json'),
                      ('repaired_linear_same_input', 'ncnn-repaired-same-input-gemv.json')]:
  path = HERE / filename
  record[key] = read(path) if path.exists() else {'status': 'pending'}
steady = HERE / 'emulator-candidate45zj-chain-device-steady.json'
if steady.exists():
  controls = read(steady)['runtime_reports']['controlsd']
  record['final_controls'] = {key: controls.get(key) for key in
    ('p99_work_ms', 'p99_complete_cycle_ms', 'mean_cycle_hz', 'scheduled_deadline_misses',
     'cycle_deadline_misses', 'skipped_slots_total', 'p99_stages_ms')}
  record['final_controls']['timing_qualification'] = 'failed'
(HERE/'validation-summary.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'experiments':record['emulator_experiments'],'controls_ab':comparison},indent=2))
