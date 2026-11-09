# Candidate45 reproduction

Run from `/Users/sergios/sunnypilot` on `codex/fold3-android-port`.
Keep the pinned assets and original model goldens described in the
[Android build guide](../../README.md). Do not regenerate goldens to clear
a numerical failure. The tolerances remain `rtol=1e-3`, `atol=1e-4`.

The source-built MNN engine is pinned by `android/scripts/build_mnn.py` and
patched by `android/patches/mnn-half-cast.patch`. Its build stamp binds the
source archive, patch, recipe, NDK and API. Rebuild the engine before converting
graphs; the converter rejects a stale native stamp.

```sh
.venv/bin/python android/scripts/build_mnn.py
.venv/bin/python android/scripts/prepare_accel_models.py --engine mnn --mnn-patched
```

The second command must pass all 795 original-reference comparisons on the
host. These verify the derived ONNX graphs; Android engine acceptance is a
separate gate. Original model files and derived graphs have separate hashes.

```sh
cd android
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python \
  ./gradlew --no-daemon --console plain \
  -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true \
  :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
```

Candidate45 is the immutable copy of candidate45h in
`android/build/candidates/candidate45`. Its APK SHA-256 is
`9868e5b9ca7d24feaefd9acfbcdabaf4f5226e85063e03fffec4400aecc1a859`;
the instrumentation APK SHA-256 is
`b8336ebab92c8bfbe076f45c42c462acb5fc1f35c0072453dc39ef07c7ab3945`.
A rebuild must be hashed and archived separately; matching source does not
establish byte-identical APKs.

Install both artifacts on `RFCR81Y1TBJ` and `emulator-5560`:

```sh
/Users/sergios/Library/Android/sdk/platform-tools/adb -s RFCR81Y1TBJ \
  install -r android/build/candidates/candidate45/app-debug.apk
/Users/sergios/Library/Android/sdk/platform-tools/adb -s RFCR81Y1TBJ \
  install -r android/build/candidates/candidate45/app-debug-androidTest.apk
.venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
  --serial RFCR81Y1TBJ --output /tmp/candidate45-installed-fold.json
```

The application ID is `com.samsung.sunnypilot.fold3`; Java class names retain
`ai.sunnypilot.fold3`. The old `ai.sunnypilot.fold3` application may also be
installed. Pass the same application ID to both the experiment wrapper and
each child validator supporting `--package`. The wrapper rejects a mismatched
model-validation target before starting an experiment.

Use new report paths for every repeat. A model test that exits with a numerical
failure is expected to return nonzero; preserve its report and cleanup result.

```sh
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --report /tmp/candidate45-mnn-isolation.json -- \
  .venv/bin/python android/scripts/validate_models.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --profile mnn-opencl-buffer --timeout 600 \
  --report /tmp/candidate45-mnn.json
```

`mnn-opencl` selects image memory; `mnn-opencl-buffer` selects buffers. The
diagnostic profiles `trace-mnn-image` and `trace-mnn-buffer:32` capture bounded
windows of actual GPU tensors. Trace offsets range from 0 to 4096; windows
contain at most 32 tensors and 128 MiB. Trace runs serialize readback and cannot
provide accelerator performance or qualification evidence.

After cleanup, collect a trace before another trace overwrites its application
directory:

```sh
.venv/bin/python android/scripts/collect_accel_trace.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --report /tmp/candidate45-trace.json --artifacts /tmp/candidate45-traces
.venv/bin/python android/reports/full-model-controls-20261101/compare_trace_window.py \
  --candidate android/build/candidates/candidate45 \
  --trace /tmp/candidate45-traces/accel-trace-driving_supercombo-buffer \
  --output /tmp/candidate45-road-trace-comparison.json
```

Use the immutable candidate graph when comparing later. Tensor names must map
exactly to the derived graph. Pre-Cast FP32 values remain diagnostic and are
never rounded afterward merely to remove a failed comparison. Acceptance uses
the full untapped model, its recurrent sequences and the pinned replay.

The `fold-candidate45-validation-driver.json` and
`emulator-candidate45-validation-driver.json` inventories retain exact command
arrays for instrumentation, deterministic process-chain input, lifecycle,
faults, IPC and MetaDrive. The later emulator retry inventory, if present,
retains its own commands and results. To repeat, copy an inventory to new paths,
reset its rows to `pending`, and give every command a new report/output path.
Run its `command` arrays without a shell. `run_validation.py` resumes pending
rows and stops after an isolation failure.

The wrapper stops both app variants and tests, checks camera release and camera
provider stability, removes new ADB forwards and requires 30 continuous seconds
of NORMAL thermal status with skin temperature at or below 32 C. An emulator
without a skin sensor still requires NORMAL status and 30 seconds. It verifies
the installed APK before, at the start of, and after each experiment. Do not
disable MODERATE pauses or critical aborts. Cooling optimization is deferred.

Use the dedicated ARM64 API35 AVD `sunnypilot_fold3_candidate44_api35`, preserving
its 12 GiB data partition. The initial `-gpu host` attempt is retained. The
separate `-gpu swiftshader` retry uses the same AVD data and APKs. A System UI
ANR dialog obscured the driving activity; dismissing its observed Wait button
restored frame delivery. The retry2 inventory starts after that intervention
and retains new reports for every check. Neither renderer change nor dialog
dismissal is claimed as an application GPU fix. Boot and permissions are documented in
[the previous reproduction guide](../npu-recovery-20261101/reproduce.md).
The MetaDrive commands must use `.venv/bin/python` directly: resolving its
symlink loses the environment's dependencies. Emulator tests establish
correctness/recovery, not phone accelerator performance.

```sh
env SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk \
  CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
  PATH=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin:$PATH \
  .venv/bin/python -m unittest discover -s android/tests
```

`controls_affinity_experiment.py` is an optional debug simulator diagnostic,
not a normal runtime setting. It changes only the worker's own allowed CPU
subset and removes its configuration in `finally`. This delivery retains the
rejected requests and makes no affinity performance claim.

Rebuild the evidence summaries after validation:

```sh
.venv/bin/python android/reports/full-model-controls-20261101/build_summary.py
```

Stop applications, verify camera release and forwarding cleanup, and stop the
dedicated emulator after testing. TX is compiled out, USB transport is IN-only,
and engagement remains unconditionally disabled in every configuration here.
