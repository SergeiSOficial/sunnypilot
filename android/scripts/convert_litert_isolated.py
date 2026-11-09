#!/usr/bin/env python3
"""Run onnx2tf on a disposable copy and retain source/conversion provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def digest(path):
  with path.open('rb') as source:
    return hashlib.file_digest(source, 'sha256').hexdigest()


def convert(source, expected, converter, output):
  source = source.resolve(strict=True)
  before = digest(source)
  if before != expected:
    raise ValueError('Source hash does not match the required SHA256')
  output.mkdir(parents=True, exist_ok=False)
  receipt = {'qualified': False, 'vehicle_tx_enabled': False, 'source': str(source),
             'source_sha256': before, 'converter': str(converter), 'status': 'error'}
  try:
    with tempfile.TemporaryDirectory(prefix='litert-source-') as temporary:
      staged = Path(temporary) / source.name
      shutil.copyfile(source, staged)
      command = [str(converter), '-i', str(staged), '-o', str(output / 'converted'),
                 '-tb', 'flatbuffer_direct', '-v', 'error']
      with (output / 'conversion.log').open('wb') as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
      receipt.update(command=command, returncode=completed.returncode,
                     staged_sha256_before=before, staged_sha256_after=digest(staged),
                     source_sha256_after=digest(source))
      receipt['staged_input_mutated'] = receipt['staged_sha256_after'] != before
      if receipt['source_sha256_after'] != before:
        raise RuntimeError('Source changed during conversion')
      receipt['outputs'] = {str(p.relative_to(output)): digest(p) for p in sorted((output / 'converted').rglob('*')) if p.is_file()}
      has_model = any(name.endswith('.tflite') for name in receipt['outputs'])
      receipt['status'] = 'converted' if completed.returncode == 0 and has_model else 'failed'
  except Exception as exc:
    receipt['error'] = f'{type(exc).__name__}: {exc}'
  (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
  return receipt


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--source', type=Path, required=True)
  parser.add_argument('--source-sha256', required=True)
  parser.add_argument('--converter', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  receipt = convert(args.source, args.source_sha256, args.converter, args.output)
  print(json.dumps(receipt, indent=2))
  return 0 if receipt['status'] == 'converted' else 1


if __name__ == '__main__':
  raise SystemExit(main())
