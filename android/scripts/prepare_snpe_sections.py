#!/usr/bin/env python3
"""Original section references and explicit SNPE layout transport diagnostics."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
from convert_snpe_linux import digest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'android/app/src/main/python'))
from fold3_sequences import initial_inputs


def required_inputs(model):
  needed = {value.name for value in model.graph.output}
  for node in reversed(model.graph.node):
    if needed.intersection(node.output): needed.update(node.input)
  return {value.name for value in model.graph.input if value.name in needed}


def prepare(sections, output):
  output.mkdir(parents=True, exist_ok=True)
  report = {'qualified': False, 'vehicle_tx_enabled': False, 'cases': {}}
  sessions = {name: ort.InferenceSession(str(sections/(name+'.onnx')), providers=['CPUExecutionProvider'])
              for name in ('vision', 'policy')}
  active = {name: required_inputs(onnx.load(sections/(name+'.onnx'))) for name in sessions}
  for seed in (42, 7, 123):
    values = initial_inputs(ROOT/'android/app/build/model-reference', 'driving_supercombo', seed)
    for section, session in sessions.items():
      dtypes = {'tensor(float)': np.float32, 'tensor(float16)': np.float16, 'tensor(uint8)': np.uint8}
      inputs = {i.name: values[i.name].reshape(i.shape).astype(dtypes[i.type], copy=False)
                for i in session.get_inputs()}
      outputs = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs)))
      folder = output/(section+'-'+str(seed)); folder.mkdir()
      np.savez(folder/'reference.npz', **outputs)
      entry = {'section': section, 'seed': seed, 'inputs': {}, 'outputs': {}, 'section_sha256': digest(sections/(section+'.onnx')),
               'omitted_unused_inputs': sorted(set(inputs)-active[section])}
      for index, (name, array) in enumerate(inputs.items()):
        if name not in active[section]: continue
        # Explicit mappings confirmed against DLC-info tensor contracts. Unit axes
        # still move; never infer a layout from element count alone.
        axes = {'new_img': (0,2,3,1), 'state_img_q': (0,2,3,4,1),
                'state_desire_q': (0,2,1), 'state_feat_q': (0,2,1)}.get(name)
        if name == 'state_img_q' and array.ndim == 4: axes = (0,2,3,1)
        converted = array.transpose(axes) if axes else array
        path = folder/(str(index)+'.raw'); converted.astype(np.float32).tofile(path)
        entry['inputs'][name] = {'file': path.name, 'sha256': digest(path), 'axes': axes,
                                 'original_shape': list(array.shape), 'dlc_shape': list(converted.shape)}
      for name, array in outputs.items():
        axes = {'next_state_img_q': (0,2,3,4,1), 'next_state_desire_q': (0,2,1),
                'next_state_feat_q': (0,2,1)}.get(name)
        if name == 'next_state_img_q' and array.ndim == 4: axes = (0,2,3,1)
        entry['outputs'][name] = {'original_shape': list(array.shape), 'axes': axes,
                                  'dlc_shape': list(array.transpose(axes).shape if axes else array.shape)}
      report['cases'][folder.name] = entry; values.update(outputs)
  (output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--sections', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args(); prepare(args.sections, args.output)
