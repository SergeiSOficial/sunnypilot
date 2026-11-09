#!/usr/bin/env python3
"""Explicit CPU diagnostics for DLC conversion errors, never GPU qualification."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from setup_snpe_linux import environment
from convert_snpe_linux import digest


def validate(root, models, fixtures, output):
  if output.exists(): raise ValueError('Preserve earlier evidence; use a new output directory')
  output.mkdir(parents=True)
  sdk, env = environment(root)
  manifest = json.loads((fixtures/'manifest.json').read_text())
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'runtime': 'explicit diagnostic CPU', 'results': {}}
  for name, case in manifest['cases'].items():
    folder = output/name; folder.mkdir()
    listing = []
    for tensor, contract in case['inputs'].items():
      path = fixtures/name/contract['file']
      if digest(path) != contract['sha256']: raise ValueError('Fixture hash mismatch')
      listing.append(tensor+':='+str(path.resolve()))
    inputs = folder/'inputs.txt'; inputs.write_text(' '.join(listing)+'\n')
    model = models/(case['section']+'.dlc')
    command = [str(sdk/'bin/x86_64-linux-clang/snpe-net-run'), '--container', str(model.resolve()),
               '--input_list', str(inputs.resolve()), '--runtime_order', 'cpu',
               '--set_output_tensors', case['section']+' '+','.join(case['outputs']), '--output_dir', str((folder/'output').resolve())]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    (folder/'run.log').write_text(result.stdout+result.stderr)
    entry = {'command': command, 'returncode': result.returncode, 'dlc_sha256': digest(model),
             'section_sha256': case['section_sha256'], 'outputs': {}, 'status': 'execution_failed'}
    if result.returncode == 0:
      with np.load(fixtures/name/'reference.npz') as reference:
        for tensor, contract in case['outputs'].items():
          path = folder/'output/Result_0'/(tensor+'.raw')
          if not path.exists():
            entry['outputs'][tensor] = {'passed': False, 'error': 'missing output'}; continue
          value = np.fromfile(path, np.float32).reshape(contract['dlc_shape'])
          if contract['axes']: value = value.transpose(np.argsort(contract['axes']))
          expected = reference[tensor].astype(np.float32)
          mismatch = ~np.isclose(value, expected, rtol=1e-3, atol=1e-4) | ~np.isfinite(value)
          entry['outputs'][tensor] = {'passed': not bool(mismatch.any()), 'mismatches': int(mismatch.sum()),
                                    'max_abs_error': float(np.abs(value-expected).max())}
      entry['status'] = 'parity_pass' if all(v['passed'] for v in entry['outputs'].values()) else 'parity_fail'
    report['results'][name] = entry
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(name, entry['status'], entry['outputs'], flush=True)
  return 0 if report['results'] and len(report['results']) == len(manifest['cases']) and all(v['status']=='parity_pass' for v in report['results'].values()) else 1


if __name__ == '__main__':
  p = argparse.ArgumentParser(description=__doc__)
  for name in ('root', 'models', 'fixtures', 'output'): p.add_argument('--'+name, type=Path, required=True)
  args = p.parse_args(); raise SystemExit(validate(args.root, args.models, args.fixtures, args.output))
