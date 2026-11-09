# Candidate45zg reproduction

Run from `/Users/sergios/sunnypilot`, branch `codex/fold3-android-port`. All accelerated variants in this delivery are diagnostic. TX remains compiled out and physical engagement disabled. Keep the candidate35/40/44/45ze/45zf archives and all failing reports.

## Build

```sh
.venv/bin/python android/scripts/build_mnn.py
.venv/bin/python android/scripts/build_ncnn.py
# Preserve strict MNN road and the original reference models first.
.venv/bin/python android/scripts/prepare_accel_models.py --engine mnn --mnn-patched
.venv/bin/python android/scripts/prepare_accel_models.py --engine ncnn --ncnn-explicit-casts
# Separate native-convolution driver variant; original hashes/tolerances unchanged.
.venv/bin/python android/scripts/prepare_accel_models.py --engine mnn --mnn-patched \
  --model dmonitoring_model --mnn-native-convolutions \
  --destination android/build/mnn-native-conv-diagnostic
.venv/bin/python android/reports/gpu-runtime-recovery-20261103/stage_candidate45zf.py
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python \
  android/gradlew -p android --no-daemon --console plain \
  -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
.venv/bin/python android/reports/gpu-runtime-recovery-20261103/archive_gpu_candidate.py FRESH-CANDIDATE
```

The native libraries and graph conversions are separately hashed; build recipes reject stale patch identities. Use a fresh archive name rather than overwriting candidate45zg. Its APK SHA-256 is `28c7d37923f7b2537c56181a85bc406f65011030a37205f2bbc1524a0e10a3f7`. Appending the trace-only batch adapter after the APK build changes no packaged code; its host tests are retained separately.

## Host checks

```sh
env CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ \
  SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest discover -s android/tests
```

186 checks passed in `host-tests-final2.log`. The initial SDK27.2 link failure is preserved in the earlier log; use the installed SDK26.5 for these host C++ fixtures.

## Install and verify each device sequentially

```sh
/Users/sergios/Library/Android/sdk/platform-tools/adb -s RFCR81Y1TBJ install -r android/build/candidates/candidate45zg/app-debug.apk
/Users/sergios/Library/Android/sdk/platform-tools/adb -s RFCR81Y1TBJ install -r android/build/candidates/candidate45zg/app-debug-androidTest.apk
.venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
  --candidate android/build/candidates/candidate45zg --serial RFCR81Y1TBJ --output FRESH-INSTALLED.json
```

Repeat with `emulator-5560` only after the physical experiments finish and clean up. Emulator boot recipe:

```sh
/Users/sergios/Library/Android/sdk/emulator/emulator \
  -avd sunnypilot_fold3_candidate45_api35 -port 5560 \
  -no-window -no-audio -no-boot-anim -no-snapshot \
  -gpu host -feature -Vulkan -memory 4096 -cores 4 \
  -camera-back emulated -camera-front emulated
```

## Device tests and diagnostics

The three reviewed inventories in this folder contain the exact commands: `fold-candidate45zg-validation-driver.json`, `fold-candidate45zg-accuracy-validation-driver.json`, and `emulator-candidate45zg-validation-driver.json`. Run them sequentially with `android/reports/full-model-controls-20261101/run_validation.py INVENTORY`. They are stateful evidence: clone completed inventories to fresh report paths and reset statuses before rerunning, preserving earlier failures. Every experiment stops app/test processes, checks camera release, removes its forwarding and measures the required 30 continuous NORMAL seconds with skin <=32C. Cooling optimization is deferred; pauses and aborts remain active.

Isolated Vulkan driver trace, using a fresh evidence path:

```sh
.venv/bin/python android/scripts/probe_ncnn.py --serial RFCR81Y1TBJ \
  --adb /Users/sergios/Library/Android/sdk/platform-tools/adb \
  --trace-count 64 --trace-skip 400 --report FRESH-PROBE.json
.venv/bin/python android/scripts/compare_ncnn_trace.py FRESH-PROBE.json
```

Shell probes hash their own provider independently of the installed APK. Intermediate tracing retains buffers and performs serialized readbacks, so its times cannot qualify accelerator speed. `diagnose_mnn_head.py` reproduces the same-captured-input strict-GEMV comparison without touching the device.

Only after all initial comparisons pass, use `run_accuracy_parts.py FRESH-CANDIDATE --models driver road` to run all 795 original-reference comparisons. Do not run qualified accelerator timing or combined 30-second/2-minute/30-minute workloads until complete numerical, camera freshness, timing and watchdog gates pass. Emulator fixtures cannot qualify physical accelerator or vehicle performance.

Generate the final artifact-bound report with `.venv/bin/python android/reports/gpu-runtime-recovery-20261103/summarize_results.py`; it rejects incomplete inventories and mixed APK identities. Stop the emulator after final process/camera/forwarding cleanup.

`run_final_devices.py` is the sequential delivery runner: it requires completed Fold functional isolation records, runs the two final accelerator rejection checks, verifies an idle Fold, boots the dedicated emulator, installs and verifies the same archived APKs, runs its reviewed inventory, then records cleanup and shuts the emulator down. It refuses to start alongside an existing emulator.

The three-seed native MNN failures block its recurrent/replay and performance qualification. The historical 795 passing comparisons belong to candidate45ze, not this APK. Original model hashes are cross-checked against that archive in `original-model-hashes.json`.
