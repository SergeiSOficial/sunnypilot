"""Diagnostic only: measure the controls worker on the phone's big CPU cluster."""
from pathlib import Path
import json
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from experiment import Device
adb='/Users/sergios/Library/Android/sdk/platform-tools/adb'
serial='RFCR81Y1TBJ'
package='com.samsung.sunnypilot.fold3'
def command(*args):
  return subprocess.check_output([adb,'-s',serial,*args],text=True,timeout=30)
configuration='files/controls-affinity-diagnostic.json'
subprocess.run([adb,'-s',serial,'shell','-T','run-as',package,'tee',configuration],
               input='[4,5,6,7]\n',text=True,stdout=subprocess.DEVNULL,check=True,timeout=30)
fixture=None
try:
  Device(serial,adb,package).start_runtime(True,full_chain=True)
  fixture=subprocess.Popen([sys.executable,'-m','openpilot.tools.sim.android_chain_validation','--seconds','30','--device-replay','--steady-inputs','--serial',serial,'--adb',adb,'--report',sys.argv[1],'--package',package])
  limit=time.monotonic()+90
  while True:
    r=json.loads(command('exec-out','run-as',package,'cat','files/controlsd.json'))
    manager=json.loads(command('exec-out','run-as',package,'cat','files/manager.json'))
    current=manager.get('processes',{}).get('Controls',{}).get('pid')
    if current and r.get('pid') == current and r.get('android_thread_priority') == -4 and r.get('work_samples',0)>=10:
      break
    if fixture.poll() is not None or time.monotonic()>=limit:
      raise TimeoutError('Fresh running controls worker did not initialize')
    time.sleep(.5)
  affinity=r.get('android_controls_affinity')
  if not affinity or affinity.get('after')!=[4,5,6,7] or not affinity.get('diagnostic_only'):
    raise RuntimeError('Self-only diagnostic affinity was not applied')
  Path(sys.argv[1]+'.affinity.json').write_text(json.dumps({'diagnostic_only':True,
      'pid':r['pid'],'tid':r['android_worker_tid'],**affinity},indent=2)+'\n')
  raise SystemExit(fixture.wait(timeout=180))
finally:
  command('shell','run-as',package,'rm','-f',configuration)
  if fixture is not None and fixture.poll() is None:
    fixture.terminate()
    try: fixture.wait(timeout=15)
    except subprocess.TimeoutExpired:
      fixture.kill();fixture.wait(timeout=10)
