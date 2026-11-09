# Candidate38 reproduction

Branch: `codex/fold3-android-port`. The archived application is
`android/build/candidates/candidate38/app-debug.apk`, SHA-256
`e00e67684a50ed5cd23cd8f55d0bcd7a0337dec049774012feeccbd665b8959f`.
The instrumentation APK is `app-debug-androidTest.apk`, SHA-256
`98d31adfd4ef7679e70a5fb51083bff55380a0c1080336e9a63278e531efa9fe`.
TX is compiled out; physical engagement remains disabled. Preserve the original
model assets and goldens. QNN graphs below are diagnostic exports.

## Build and host validation

Use the pinned preparation commands in [the Android instructions](../../README.md).
Run from the repository root:

```sh
export ANDROID_HOME="$HOME/Library/Android/sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk

.venv/bin/python -m unittest discover -s android/tests
.venv/bin/python -m unittest \
  opendbc.car.hyundai.tests.test_hyundai \
  opendbc.safety.tests.test_hyundai \
  opendbc.safety.tests.test_hyundai_canfd \
  openpilot.tools.sim.tests.test_android_protocol
.venv/bin/python android/scripts/validate_host_math.py --report /tmp/fold3-host-parity.json
.venv/bin/python android/scripts/prepare_qnn.py

cd android
env FOLD3_BUILD_PYTHON="$PWD/../.venv/bin/python" ./gradlew \
  --no-daemon --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true \
  :app:cleanPackageDebug :app:assembleDebug :app:assembleDebugAndroidTest \
  :app:testDebugUnitTest
cd ..
```

The selected convolution workgroup is 128 lanes. Activation, division and other
ordered custom kernels retain their prior maximum of 32. Candidate39's larger
workgroups regressed and were reverted. A new build must receive its own hash;
APK signing/build metadata can change bytes. The [artifact record](candidate38-artifact.json)
contains source and packaged native hashes, including the pinned QNN archives.
Host script hashes describe the delivery tools; diagnostic graph exports are
separately hashed in their manifests and trace reports.

## Install

Finish the current experiment and cleanup before installing. Keep the emulator
stopped during phone performance measurements. Install the same archived APKs
on `RFCR81Y1TBJ` and, later, `emulator-5560`:

```sh
adb -s RFCR81Y1TBJ shell am force-stop com.samsung.sunnypilot.fold3
adb -s RFCR81Y1TBJ install -r -t -g android/build/candidates/candidate38/app-debug.apk
adb -s RFCR81Y1TBJ install -r -t android/build/candidates/candidate38/app-debug-androidTest.apk
adb -s RFCR81Y1TBJ shell pm path com.samsung.sunnypilot.fold3
```

Run `sha256sum` on the returned installed path and repeat for the `.test` package.
The experiment wrapper also verifies the installed application hash before and
after every run. It stops both packages, checks camera release, removes only
experiment-created forwarding and requires 30 continuous NORMAL thermal seconds
with phone skin temperature at or below 32°C before the child command starts.

## Model diagnostics and full gate

Example asynchronous event profile, including compilation, transfers, GPU events
and output readback without per-kernel waits:

```sh
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --report /tmp/driver-profile-isolation.json -- \
  .venv/bin/python android/scripts/validate_models.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --profile profile-cl-driver --report /tmp/driver-profile.json
```

Use `profile-cl-road` for the other model. These five-sample profiles cannot
qualify a backend. The shared validator returns nonzero/blocked because no full
qualification result is produced; inspect the model's sampled parity and the
separate isolation result. Serialized tensor checkpoints are excluded from
performance comparisons.

Use `host-fusion-cl-driver` and `host-fusion-cl-road` instead for all three seeds,
32-frame recurrent sequences, the pinned 60-frame replay and 100 timing samples.
Their numerical and performance results are separate. A failed timing gate
returns nonzero even when every numerical comparison passes. Only a complete
two-model run passing every gate can write live qualification, bound to the
actual APK, implementation, model/reference assets, OpenCL library and device.

Generate the lossless QNN image-layout diagnostic, then collect its trace through
the same experiment wrapper:

```sh
.venv/bin/python android/scripts/model_trace.py --portable-host --float-image-ops \
  --model both --limit 35 --output-directory android/build/qnn-trace-float-images-rank4
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --report /tmp/qnn-trace-isolation.json -- \
  .venv/bin/python android/scripts/collect_model_trace.py \
  --serial RFCR81Y1TBJ --profile trace-qnn \
  --trace-directory android/build/qnn-trace-float-images-rank4 \
  --artifact-directory /tmp/qnn-trace-artifacts --report /tmp/qnn-trace.json
```

Trace completion is not numerical success. Inspect first divergence, unexpected
non-finite values, final outputs and the actual provider assignment. Tapped
graphs can change fusion and partitioning. QNN remains diagnostic and HTP work
remains deferred.

## Camera, runtime, emulator and simulator

The [phone command inventory](fold-candidate38-validation-driver.json) and
[emulator command inventory](emulator-candidate38-validation-driver.json) retain
the exact ordered commands and results, including every isolation report. Each
`command` array can be passed directly to Python's `subprocess.run` without a
shell. Run sequentially, stop if isolation fails, and preserve failed reports.
The inventories include four matched camera profiles with continuous GLES
readback, separate real emulator camera timing, deterministic process-chain
inputs, cancellation/staleness, 22 instrumentation tests, lifecycle/display
recovery, numerical and USB rejection faults, thermal protection and MetaDrive.

Start the dedicated ARM64 Android 15 emulator only after phone measurements:

```sh
"$ANDROID_HOME/emulator/emulator" -avd sunnypilot_fold3_api35 -port 5560 \
  -no-window -no-audio -no-boot-anim -no-snapshot -gpu host \
  -memory 4096 -cores 2 -camera-back emulated -camera-front emulated
```

Wait for `sys.boot_completed=1`, install both archived APKs, verify hashes, then
execute the emulator inventory. Stop the app and test package, verify no camera
owner or experiment forwards remain, and shut down with
`adb -s emulator-5560 emu kill` after validation.

Do not advance from short combined runs to two minutes or 30 minutes until valid
fresh independent camera inputs, numerical accuracy and timing gates pass.
Emulator/MetaDrive success cannot qualify independent rear sensors, mounted
calibration, physical Panda firmware/CAN/harness behavior or road readiness.
