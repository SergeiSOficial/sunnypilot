"""Diagnostic only: measure the controls worker on the phone's big CPU cluster."""
from pathlib import Path
import json
import subprocess
import sys
import time
adb='/Users/sergios/Library/Android/sdk/platform-tools/adb'
serial='RFCR81Y1TBJ'
package='com.samsung.sunnypilot.fold3'
def command(*args):
  return subprocess.check_output([adb,'-s',serial,*args],text=True,timeout=30)
# The fixture supplies CarParams. Start it before waiting for controls to run.
fixture=subprocess.Popen([sys.executable,'-m','openpilot.tools.sim.android_chain_validation','--seconds','30','--device-replay','--steady-inputs','--serial',serial,'--adb',adb,'--report',sys.argv[1],'--package',package])
try:
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
  pid=str(r['pid'])
  tid=str(r['android_worker_tid'])
  if not tid.isdecimal() or int(tid) <= 0:
    raise ValueError('Missing recorded controls native thread ID')
  # The worker records its own Linux TID; do not infer ownership from names
  # or priorities inherited by parameter/reporting threads.
  status=command('shell','run-as',package,'cat',f'/proc/{pid}/task/{tid}/status')
  if ('Tgid:\t'+pid) not in status:
    raise ValueError('Recorded worker TID does not belong to current process')
  before=command('shell','run-as',package,'taskset','-p',tid)
  result=command('shell','run-as',package,'taskset','-p','f0',tid)
  after=command('shell','run-as',package,'taskset','-p',tid)
  Path(sys.argv[1]+'.affinity.json').write_text(json.dumps({'diagnostic_only':True,'startup_unpinned':True,'pid':pid,'tid':tid,'before':before,'change':result,'after':after},indent=2)+'\n')
  raise SystemExit(fixture.wait(timeout=180))
finally:
  if fixture.poll() is None:
    fixture.terminate()
    try: fixture.wait(timeout=15)
    except subprocess.TimeoutExpired:
      fixture.kill();fixture.wait(timeout=10)
