# Candidate32 reproduction commands

Run from the repository root on the Mac. Complete the pinned dependency,
submodule, native-source and model preparation in [the Android build
instructions](../../README.md) first. Model bytes, reference hashes and
`rtol=1e-3, atol=1e-4` are unchanged. These are bench builds: vehicle TX is
compiled out and physical engagement is disabled.

## Build and host checks

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
.venv/bin/python android/scripts/validate_host_math.py \
  --report /tmp/candidate32-host-reference.json
.venv/bin/ruff check android/app/src/main/python/fold3_host_math.py \
  android/app/src/main/python/fold3_model_trace.py \
  android/scripts/compare_camera_trace.py android/tests/test_host_math.py

cd android
env FOLD3_BUILD_PYTHON="$PWD/../.venv/bin/python" ./gradlew \
  --no-daemon --console plain -Pfold3CameraCompat=true \
  :app:cleanPackageDebug :app:assembleDebug :app:assembleDebugAndroidTest \
  :app:testDebugUnitTest
cd ..
```

The explicit Xcode compiler/SDK selects the working host toolchain on this Mac.
The [initial Command Line Tools failure](candidate32-host-toolchain-failure.json)
is retained. [Host results](candidate32-host-results.json) record the successful
rerun. APK rebuilds must be recorded under their own actual hashes; build and
signing timestamps can change APK bytes.

## Install the archived artifact

Wait for any running experiment to finish cleanup before installing. The
[artifact record](candidate32-artifact.json) and [installation record](candidate32-installed.json)
identify the app and instrumentation APKs used for the recorded results.

```sh
shasum -a 256 android/build/candidates/candidate32/app-debug.apk \
  android/build/candidates/candidate32/app-debug-androidTest.apk
for serial in RFCR81Y1TBJ emulator-5554; do
  adb -s "$serial" install -r -g android/build/candidates/candidate32/app-debug.apk
  adb -s "$serial" install -r -g android/build/candidates/candidate32/app-debug-androidTest.apk
done
```

## Device experiments

Every validator runs through `experiment.py`. It stops app/test processes,
checks camera release and forwarding cleanup, waits at least 30 continuous
NORMAL thermal seconds with phone skin at or below 32°C, and verifies the
installed APK hash before and after the experiment.

Example: run both complete original-model gates sequentially on the phone.
Each gate includes seeds 42/7/123, 32 recurrent frames per seed, the pinned
60-frame camera replay and 100 timed inferences.

```sh
for model in road driver; do
  .venv/bin/python android/scripts/experiment.py \
    --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
    --report "/tmp/candidate32-$model-isolation.json" -- \
    .venv/bin/python android/scripts/validate_models.py \
    --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
    --profile "host-fusion-cl-$model" --timeout 900 \
    --report "/tmp/candidate32-$model.json"
done
```

The [phone batch](fold-candidate32-validation-driver.json) and
[emulator batch](emulator-candidate32-validation-driver.json) enumerate the
final experiments. Each linked `-isolation.json` contains the exact child
command, runtime mode and cleanup evidence. Reuse those child arguments
through the wrapper, selecting the same `--runtime` when present:

- Instrumentation: `validate_instrumentation.py --expected-tests 20` on both.
- Runtime chain: `openpilot.tools.sim.android_chain_validation --seconds 30`,
  then `--steady-inputs`, with wrapper `--runtime full-chain` on both.
- Camera/model load: `validate_vision.py --seconds 30 --samsung-multiview`
  on the phone; no wrapper runtime option.
- MetaDrive: `openpilot.tools.sim.android_validation --duration 20 --metadrive`
  with `--fault cancel`, then `--fault stale`; wrapper `--runtime simulation`
  on the emulator. Its child CLI does not accept `--package`.
- Lifecycle: `validate_lifecycle.py`, wrapper `--runtime physical` on the
  phone; child `--resize` and wrapper `--runtime simulation` on the emulator.
- ADB faults: `validate_adb_faults.py`, wrapper `--runtime simulation` on the
  emulator.
- IPC: `validate_msgq_wakeup.py` on both. Its child CLI uses `--output` and
  does not accept `--package`.
- Actual emulator camera timing: `validate_cameras.py --ids 1 --seconds 10`;
  its child CLI uses `--output`. Keep this result separate from injected inputs.
- Kernel checkpoints: `validate_models.py --profile trace-kernels --timeout 120`
  on the phone. This diagnostic deliberately cannot qualify live performance.

Do not start a sustained workload until the short numerical, freshness and
deadline gates pass. Physical camera identity, mounted calibration and the
disconnected Panda require their own hardware evidence.
