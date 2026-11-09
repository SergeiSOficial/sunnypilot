"""Stage diagnostic engines only; retain archived accurate candidate45ze."""
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[3]
assets = root / 'android/app/build/model-reference/accelerators'
native = root / 'android/build/mnn-native-conv-diagnostic'
manifest = json.loads((assets / 'manifest.json').read_text())
variant = json.loads((native / 'manifest.json').read_text())['models']['dmonitoring_model']
assert manifest['models']['dmonitoring_model']['source_sha256'] == variant['source_sha256']
assert manifest['models']['dmonitoring_model']['derived_sha256'] == variant['derived_sha256']
assert variant['original_reference']['passed']
assert variant['mnn']['native_convolutions'] and not variant['mnn']['host_conv_order']
assert variant['mnn']['host_gemv_order'] and variant['mnn']['explicit_half_boundaries']
manifest['models']['dmonitoring_model']['mnn'] = variant['mnn']
for filename in ('dmonitoring_model-mnn-io.onnx', 'mnn/dmonitoring_model.mnn'):
    shutil.copy2(native / filename, assets / filename)
(assets / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
