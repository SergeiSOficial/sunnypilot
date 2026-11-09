"""Bind final diagnostic evidence to its immutable candidate archive."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
name='candidate45zo'
artifacts=json.loads((ROOT/'android/build/candidates'/name/'artifacts.json').read_text())
rows=json.loads((HERE/f'emulator-{name}-validation-driver.json').read_text())
model=json.loads((HERE/'prefix-ncnn-seed42.json').read_text())
comparison=json.loads((HERE/'controls-comparison.json').read_text())
report={'candidate':name,'branch':'codex/fold3-android-port',
 'apk_sha256':artifacts['files']['app-debug.apk']['sha256'],
 'vehicle_tx_enabled':False,'physical_engagement_allowed':False,
 'host':{'status':'passed','tests':190,'evidence':'host-suite.log'},
 'projection_host_sections':{'status':'passed','exact_checks':12,'evidence':'projection-prefix-host.json'},
 'model_accuracy':{'status':'failed','seed':42,'parity':model['parity'],
   'evidence':'prefix-ncnn-seed42.json','remaining_seeds_and_795_suite':'not advanced after first seed failure',
   'backend_qualified':False},
 'controls':{'status':'failed','evidence':'controls-comparison.json','comparison':comparison},
 'emulator':{'groups':[{k:r[k] for k in ['name','status','isolation_status'] if k in r} for r in rows],
   'installed_hash':'candidate45zo-installed-emulator.json'},
 'fold3':{'status':'blocked','reason':'disconnected; accelerator performance and installed hash unverified'},
 'physical_gates':{'status':'blocked','reason':'unmounted phone, deferred rear lens-cover proof, disconnected Panda and vehicle'},
 'sustained_workload':{'status':'blocked','reason':'accuracy/timing gates fail; physical Fold 3 disconnected'},
 'cooling_optimization':'deferred; existing thermal protection and cleanup enforced',
 'cleanup':json.loads((HERE/'candidate45zo-final-cleanup.json').read_text()) if (HERE/'candidate45zo-final-cleanup.json').exists() else {'status':'pending'}}
(HERE/'validation-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'candidate':name,'model_accuracy':'failed','controls':'failed','emulator_groups':report['emulator']['groups']},indent=2))
