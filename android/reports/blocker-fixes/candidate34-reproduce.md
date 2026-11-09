# Candidate34 reproduction commands

Run from the repository root after the pinned source, model and native preparation
in [the Android build instructions](../../README.md). Original model bytes,
goldens and `rtol=1e-3, atol=1e-4` are unchanged. Vehicle TX is compiled out;
physical engagement remains disabled. The phone is unmounted and the Panda is
disconnected. Camera identity still needs the deferred lens-cover experiment.

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
  --report /tmp/candidate34-host-reference.json

cd android
env FOLD3_BUILD_PYTHON="$PWD/../.venv/bin/python" ./gradlew \
  --no-daemon --console plain -Pfold3CameraCompat=true \
  :app:cleanPackageDebug :app:assembleDebug :app:assembleDebugAndroidTest \
  :app:testDebugUnitTest
cd ..
```

The [artifact record](candidate34-artifact.json) binds the recorded APK to its
source hashes and native dependencies. Rebuilds must receive their own actual
hashes; build and signing timestamps can change APK bytes.

## Install the archived artifact

Finish the current experiment and its cleanup before installation. The Samsung
compatibility package is `com.samsung.sunnypilot.fold3`.

```sh
shasum -a 256 android/build/candidates/candidate34/app-debug.apk \
  android/build/candidates/candidate34/app-debug-androidTest.apk
for serial in RFCR81Y1TBJ emulator-5560; do
  adb -s "$serial" install -r -g android/build/candidates/candidate34/app-debug.apk
  adb -s "$serial" install -r -g android/build/candidates/candidate34/app-debug-androidTest.apk
  installed=$(adb -s "$serial" shell pm path com.samsung.sunnypilot.fold3 | tr -d '\r' | sed 's/^package://')
  adb -s "$serial" shell sha256sum "$installed"
done
```

Start the dedicated emulator after phone performance measurements and cleanup:

```sh
"$ANDROID_HOME/emulator/emulator" -avd sunnypilot_fold3_api35 -port 5560 \
  -no-window -no-audio -no-boot-anim -no-snapshot -gpu host \
  -memory 4096 -cores 2 -camera-back emulated -camera-front emulated
```

Wait for `adb -s emulator-5560 shell getprop sys.boot_completed` to return `1`
before installation. Keep emulator rendering load out of the phone measurements.

## Isolated device experiments

Every validator runs through `experiment.py`. It stops the app and tests,
checks camera release, removes only experiment-created forwarding, waits for
at least 30 continuous NORMAL thermal seconds with phone skin at or below 32°C,
and verifies the installed APK hash before and after the child. Reports record
validation and experiment isolation separately.

Run both original reference model gates sequentially on the phone. Each includes
seeds 42/7/123, 32 recurrent frames per seed, the pinned 60-frame camera replay
and 100 timed inferences:

```sh
for model in road driver; do
  .venv/bin/python android/scripts/experiment.py \
    --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
    --report "/tmp/candidate34-reference-$model-isolation.json" -- \
    .venv/bin/python android/scripts/validate_models.py \
    --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
    --profile "host-fusion-cl-$model" --timeout 900 \
    --report "/tmp/candidate34-reference-$model.json"
done
```

The `host-fusion-cl-mad-road` and `host-fusion-cl-mad-driver` profiles are separate
native multiply-add diagnostics. They cannot write live backend qualification,
even if a numerical comparison passes. The reference profiles keep ordered FMA
and the original rounding boundaries.

Measure continuous conversion/readback for each bounded camera profile:

```sh
for profile in record preview record-720p record-depth2; do
  .venv/bin/python android/scripts/experiment.py \
    --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
    --report "/tmp/candidate34-camera-$profile-isolation.json" -- \
    .venv/bin/python android/scripts/validate_cameras.py \
    --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
    --ids 0,2,1 --logical 20 --samsung-multiview --seconds 12 \
    --camera-profile "$profile" --continuous-readback \
    --output "/tmp/candidate34-camera-$profile.json"
done
```

Continuous readback exercises GPU conversion, fences, PBO readback, owned buffers
and publication timing. It does not qualify the Samsung composite as two physical
rear sensors or enable vehicle inputs. Live capture retains the default profile.

Run the sources-only complete process chain, using on-device deadlines to remove
per-packet ADB transport jitter. Derived vehicle/planning/control messages must
come from the supervised processes:

```sh
.venv/bin/python android/scripts/experiment.py \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --runtime full-chain --report /tmp/candidate34-chain-isolation.json -- \
  .venv/bin/python -m openpilot.tools.sim.android_chain_validation \
  --serial RFCR81Y1TBJ --package com.samsung.sunnypilot.fold3 \
  --seconds 30 --device-replay --steady-inputs \
  --report /tmp/candidate34-chain.json
```

Repeat without `--steady-inputs` for factory-ACC cancellation and stale-input
phases, and on `emulator-5560`. The debug replay is bounded, authenticated to
the app/ADB shell, isolated from physical parameters and restricted to synthetic
sources. Disconnecting it clears active simulator state.

Additional child commands, always through the same wrapper:

- Instrumentation: `android/scripts/validate_instrumentation.py --expected-tests 21`;
  its text log uses `--output`, JSON uses `--report`.
- Lifecycle/display: `android/scripts/validate_lifecycle.py`, wrapper
  `--runtime simulation` on both devices. Add `--resize` only on the dedicated
  emulator; display overrides are intentionally rejected on the physical phone.
- ADB faults: `android/scripts/validate_adb_faults.py`, wrapper
  `--runtime simulation` on both devices.
- Combined phone camera/model/UI load: `android/scripts/validate_vision.py
  --seconds 30 --samsung-multiview`, no wrapper runtime option.
- Emulator thermal fault handling: `android/scripts/validate_thermal.py`, no
  wrapper runtime option. Thermal overrides are restricted to the emulator.
- MetaDrive: `-m openpilot.tools.sim.android_validation --duration 20 --metadrive
  --fault cancel`, then `--fault stale`; wrapper `--runtime simulation` on the
  emulator. The child does not accept `--package`.
- IPC: `android/scripts/validate_msgq_wakeup.py` on both devices; child uses
  `--output` and does not accept `--package`.
- Actual emulator camera timing: `android/scripts/validate_cameras.py --ids 1
  --seconds 10`; child uses `--output`. Keep actual capture-clock results separate
  from deterministic injected inputs.

Each child also needs `--serial`, `--adb` when ADB is not on PATH, its report
destination, and `--package com.samsung.sunnypilot.fold3` unless excluded above.
The batch and individual isolation records retain the exact executed commands.
Stop the emulator with `adb -s emulator-5560 emu kill` after final cleanup.

Do not advance to the 30-minute workload until numerical, camera identity,
freshness and short-run timing gates pass. Mounted calibration, physical Panda
firmware compatibility, passive fingerprint and vehicle behavior need hardware
evidence regardless of emulator or simulator results.
