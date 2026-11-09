#!/usr/bin/env python3
"""Execute section probes without CPU fallback; never qualify split perception."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import numpy as np
from convert_snpe_linux import digest


def contracts(info, label):
  table = info.split('| '+label+' Name', 1)[1].split('Total parameters:', 1)[0]
  if label == 'Input': table = table.split('| Output Name', 1)[0]
  return {name.strip(): [int(x) for x in dims.split(',')] for name, dims in
          re.findall(r'^\|\s*([^|]+?)\s*\|\s*([0-9,]+)\s*\|\s*(?:Float_32|Uint_8)', table, re.M)}


def remote_shell(command):
  """Preserve argument boundaries through adb's remote shell."""
  return ['shell', shlex.join(command)]


def main():
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('adb', 'serial'): p.add_argument('--'+name, required=True)
  for name in ('fixtures', 'converted', 'runtime', 'output'): p.add_argument('--'+name, type=Path, required=True)
  args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
  base = [args.adb,'-s',args.serial]; remote = '/data/local/tmp/fold3-c74-snpe-sections'
  def adb(*command, check=True):
    return subprocess.run(base+list(command),check=check,capture_output=True,text=True,timeout=180)
  def upload(path, target):
    adb('push',str(path),target)
    if adb('shell','sha256sum',target).stdout.split()[0] != digest(path): raise ValueError('Uploaded hash mismatch')
  report = {'qualified':False,'vehicle_tx_enabled':False,'execution_domain':'ADB shell, not app',
            'cpu_fallback_enabled':False,'rtol':1e-3,'atol':1e-4,'results':{}}
  try:
    adb('shell','mkdir','-p',remote)
    for file in ('bin/aarch64-android/snpe-net-run','lib/aarch64-android/libSNPE.so'):
      source=args.runtime/file; upload(source,remote+'/'+source.name)
    adb('shell','chmod','700',remote+'/snpe-net-run')
    manifest=json.loads((args.fixtures/'manifest.json').read_text())
    for name, case in manifest['cases'].items():
      folder=args.output/name; folder.mkdir(); section=case['section']
      info=(args.converted/(section+'.info.txt')).read_text(); inputs=contracts(info,'Input'); outputs=contracts(info,'Output')
      listing=[]
      for index,(tensor, shape) in enumerate(inputs.items()):
        entry=case['inputs'][tensor]
        if entry['dlc_shape'] != shape: raise ValueError('Unproven input layout: '+tensor)
        target=remote+'/'+str(index)+'.raw'; upload(args.fixtures/name/entry['file'],target)
        listing.append(tensor+':='+target)
      for tensor,shape in outputs.items():
        if case['outputs'][tensor]['dlc_shape'] != shape: raise ValueError('Unproven output layout: '+tensor)
      path=folder/'input-list.txt';path.write_text(' '.join(listing)+'\n');upload(path,remote+'/input-list.txt')
      model=args.converted/(section+'.dlc');upload(model,remote+'/model.dlc')
      adb('shell','rm','-rf',remote+'/output')
      # adb shell joins its arguments before passing them to the remote shell.
      # Quote the whole command so the graph and output names remain one argument.
      command=remote_shell(['env','LD_LIBRARY_PATH='+remote,remote+'/snpe-net-run','--container',remote+'/model.dlc',
               '--input_list',remote+'/input-list.txt','--set_output_tensors',section+' '+','.join(case['outputs']),'--runtime_order','gpu','--profiling_level','detailed','--output_dir',remote+'/output'])
      result=adb(*command,check=False);(folder/'run.log').write_text(result.stdout+result.stderr)
      adb('pull',remote+'/output',str(folder),check=False)
      check={'returncode':result.returncode,'requested_runtime':'gpu','dlc_sha256':digest(model),'outputs':{},
             'status':'execution_failed','actual_assignment':'pending_diag_decode','command':base+command}
      if result.returncode == 0:
        with np.load(args.fixtures/name/'reference.npz') as references:
          for tensor, contract in case['outputs'].items():
            candidates=[f for f in folder.rglob('*.raw') if f.as_posix().endswith('/Result_0/'+tensor+'.raw')]
            if len(candidates)!=1: check['outputs'][tensor]={'passed':False,'error':'missing or ambiguous output'};continue
            actual=np.fromfile(candidates[0],np.float32).reshape(contract['dlc_shape'])
            if contract['axes']: actual=actual.transpose(np.argsort(contract['axes']))
            expected=references[tensor].astype(np.float32)
            mismatch=~np.isclose(actual,expected,rtol=1e-3,atol=1e-4) | ~np.isfinite(actual)
            check['outputs'][tensor]={'passed':not bool(mismatch.any()),'mismatches':int(mismatch.sum()),
                                      'elements':int(actual.size),'max_abs_error':float(np.abs(actual-expected).max())}
        check['status']='parity_pass' if all(v['passed'] for v in check['outputs'].values()) else 'parity_fail'
      report['results'][name]=check;(args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
      print(name,check['status'],check['outputs'],flush=True)
  finally:
    adb('shell','rm','-rf',remote)
    report['remote_directory_removed']=adb('shell','test','!','-e',remote,check=False).returncode==0
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
  return 0 if report['remote_directory_removed'] and len(report['results'])==6 and all(v['status']=='parity_pass' for v in report['results'].values()) else 1


if __name__=='__main__':raise SystemExit(main())
