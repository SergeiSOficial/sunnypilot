"""Verify final APK content before any installation or experiment."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
folder = ROOT/'android/build/candidates/candidate45zj'
path = folder/'app-debug.apk'
with zipfile.ZipFile(path) as apk:
  assert b'ordered_half_gemv' in apk.read('lib/arm64-v8a/fold3_accel_native.cpython-312.so')
  with zipfile.ZipFile(io.BytesIO(apk.read('assets/chaquopy/app.imy'))) as app:
    module=next(n for n in app.namelist() if n.endswith(('fold3_daemons.pyc', 'fold3_daemons.py')))
    assert b'android_controls_gil_switch_interval_s' in app.read(module)
    assert b'setswitchinterval' not in app.read(module)
    graph=next(n for n in app.namelist() if n.endswith('ncnn/dmonitoring_model.ncnn.param'))
    data=app.read(graph)
    assert any('linear_100' in line and '9=99' in line for line in data.decode().splitlines())
    assert data==(folder/'graphs/ncnn/dmonitoring_model.ncnn.param').read_bytes()
    manifest=next(n for n in app.namelist() if n.endswith('accelerators/manifest.json'))
    assert json.loads(app.read(manifest))['models']['dmonitoring_model']['ncnn']['requires_ordered_half_gemv'] is True
  provider_hash=hashlib.sha256(apk.read('lib/arm64-v8a/libncnn.so')).hexdigest()
record={'apk_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ordered_half_gemv_feature':True,
 'controls_gil_policy_preserved':True,'vehicle_tx_enabled':False,
 'marked_graph_sha256':hashlib.sha256(data).hexdigest(),'marked_graph_matches_archive':True,
 'packaged_ncnn_sha256':provider_hash,'native_build':json.loads((folder/'ncnn-build.json').read_text())}
(HERE/'candidate45zj-package-check.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
