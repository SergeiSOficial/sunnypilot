# Candidate45zd reproduction

Run from `/Users/sergios/sunnypilot` on `codex/fold3-android-port`.
Preserve existing candidate archives and experiment reports. This APK keeps
vehicle transmission compiled out and physical engagement disabled.

## Build

Use the matched Xcode compiler/SDK for host native checks. The complete
177-test run is recorded in `candidate45zd-all-host.log`.

```sh
env CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
  SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest discover -s android/tests
env CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
  SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest opendbc.safety.tests.test_hyundai \
  opendbc.car.hyundai.tests.test_hyundai \
  openpilot.tools.sim.tests.test_android_protocol openpilot.tools.sim.tests.test_metadrive_bridge
```

The source archive and MNN patch are checksum bound by `build_mnn.py`.
Conversion preserves original FP16 boundaries, original model hashes and
`rtol=1e-3`, `atol=1e-4`. It also runs all 795 original-reference comparisons
on the host; those comparisons do not qualify phone execution.

```sh
.venv/bin/python android/scripts/build_mnn.py
.venv/bin/python android/scripts/prepare_accel_models.py --engine mnn --mnn-patched
cd android
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python \
  ./gradlew --no-daemon --console plain \
  -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true \
  :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
cd ..
.venv/bin/python android/reports/full-model-controls-20261101/archive_candidate.py candidate45zd
```

The archive command refuses to overwrite an existing candidate. The artifact
record retains packaged library hashes separately from unstripped build hashes.

## Install and identify

Stop any active experiment before installation. Repeat these commands for
`RFCR81Y1TBJ` and the dedicated `emulator-5560`.

```sh
FOLD3_ADB=/Users/sergios/Library/Android/sdk/platform-tools/adb
FOLD3_SERIAL=RFCR81Y1TBJ
"$FOLD3_ADB" -s "$FOLD3_SERIAL" install -r -t -g \
  android/build/candidates/candidate45zd/app-debug.apk
"$FOLD3_ADB" -s "$FOLD3_SERIAL" install -r -t -g \
  android/build/candidates/candidate45zd/app-debug-androidTest.apk
.venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
  --candidate android/build/candidates/candidate45zd --serial "$FOLD3_SERIAL" \
  --output android/reports/full-model-controls-20261101/candidate45zd-installed-fold.json
```

Use a new output name for a new attempt. A successful build or install does not
establish numerical accuracy, timing, camera identity or vehicle readiness.

## Numerical acceptance and device checks

Mac system sleep interrupted earlier USB attempts. Prefix a long-running
command with `/usr/bin/caffeinate -is` to keep its host awake only for that
command's lifetime. This changes neither persistent power settings nor phone
thermal safeguards.

```sh
.venv/bin/python android/reports/full-model-controls-20261101/run_accuracy_parts.py candidate45zd
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py \
  android/reports/full-model-controls-20261101/fold-candidate45zd-accelerator-timing-driver.json
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py \
  android/reports/full-model-controls-20261101/emulator-candidate45zd-validation-driver.json
.venv/bin/python android/reports/full-model-controls-20261101/run_validation.py \
  android/reports/full-model-controls-20261101/fold-candidate45zd-validation-driver.json
```

The reviewed inventories preserve completed evidence and skip completed rows.
Each experiment uses `android/scripts/experiment.py`: stop app/test processes,
verify camera release, remove experiment forwards, wait for 30 continuous
NORMAL seconds at skin temperature at or below 32°C, and verify the APK before
and after execution. The emulator uses NORMAL status because its temperature
sensor is virtual. MODERATE pause and critical abort remain enabled.

Keep the emulator idle or stopped during matched Fold controls timing. Run
accelerator timing only after both complete original-reference accuracy gates
pass on this APK. Advance combined visual workloads only after accuracy,
freshness, timing and watchdog prerequisites pass. Earlier failures and
partially executed stages remain retained under their actual APK identities.

After the inventories finish, generate the artifact-bound reports with the
actual successful build log. Earlier candidate build attempts remain retained.

```sh
.venv/bin/python android/reports/full-model-controls-20261101/build_latest_summary.py candidate45zd \
  --build-evidence android/reports/full-model-controls-20261101/candidate45zd-build.log \
  --model-evidence android/reports/full-model-controls-20261101/fold45zd-mnn-road-accuracy-complete.json \
  --model-evidence android/reports/full-model-controls-20261101/fold45zd-mnn-driver-accuracy-complete.json \
  --timing-evidence android/reports/full-model-controls-20261101/fold45zd-mnn-driver-timing.json \
  --timing-evidence android/reports/full-model-controls-20261101/fold45zd-mnn-road-timing.json
```

This generator preserves failed and thermally blocked stages. It cannot turn
a partial benchmark into a p99 result or transfer evidence between APKs.

## Attention diagnosis

The retained candidate45z recurrent inputs reproduce the reduction-order error
without changing the original reference:

```sh
.venv/bin/python android/reports/full-model-controls-20261101/probe_attention_actual.py \
  --candidate android/build/candidates/candidate45z \
  --trace android/build/candidates/candidate45z/trace608-seed42-frame26/accel-trace-driving_supercombo-buffer \
  --output /tmp/fold3-attention-diagnosis.json
```

The captured second attention MatMul differs in one half value. Paired FP32
accumulation with the odd-K tail agrees exactly with the host on these inputs;
sequential accumulation reproduces the error. This diagnostic is separate from
complete untapped-model acceptance and supplies no performance qualification.

## Summarize the actual artifacts

After both accuracy proofs and the device inventories exist:

```sh
.venv/bin/python android/reports/full-model-controls-20261101/build_latest_summary.py candidate45zd \
  --model-evidence android/reports/full-model-controls-20261101/fold45zd-mnn-road-accuracy-complete.json \
  --model-evidence android/reports/full-model-controls-20261101/fold45zd-mnn-driver-accuracy-complete.json \
  --timing-evidence android/reports/full-model-controls-20261101/fold45zd-mnn-driver-timing.json \
  --timing-evidence android/reports/full-model-controls-20261101/fold45zd-mnn-road-timing.json
```

Timing interrupted by a thermal safeguard stays blocked; a last-call stage
snapshot is not a completed sample set. Host wall stages can include queued GPU
execution and must not be presented as individual GPU kernel event times.
