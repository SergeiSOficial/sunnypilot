# Fold 3 three-camera recovery, 6 November 2026

Branch `codex/fold3-android-port`. The APK is [candidate45-capture-budget](../../build/candidates/candidate45-capture-budget/app-debug.apk). Vehicle TX is compiled out and physical engagement is disabled. Raw receipts are in `android/build/three-camera-recovery.3N6tff`.

| Artifact | SHA-256 |
| --- | --- |
| Application | `b35357542d827ee60620caeef36a1504ce16bb68de4be08f99438e027882f0a9` |
| Instrumentation | `96876c828001aafe773f8d00589528522dce16709372ab77e134117186de6a6f` |
| Initial camera/trace APK | `22cdb497383fabf70952d911026832531435dc47beb7316b1408b03492e0fc2a` |
| Prior matched controls APK | `b820cad4c36221cdac90a8a63662c25f258370185387fc50fb7517593f24f093` |

## Camera timing

The stock Android 15 Fold 3 advertises no public concurrent ID sets and no physical groups on its public IDs. Hidden IDs 20 and 23 report physical rear groups. The provider-crash exclusions recorded in [camera history](../../CAMERAS.md) still apply. Android's supported physical-stream substitution relies on an accessible logical camera and compatible outputs. [Android Camera2 multi-camera guide](https://developer.android.com/media/camera/camera2/multi-camera)

Both previously supported direct-YUV pairs captured for eight seconds on the initial APK. These figures measure sensor-capture age using sampled clock conversion and contain no inference.

| Pair and capture size | Camera fps | Acquisition p99, ms | Delivery p99, ms |
| --- | ---: | ---: | ---: |
| Rear wide 0 + front 1, 640×480 | 29.9 / 29.8 | 86.42 / 89.31 | 87.83 / 92.18 |
| Rear ultrawide 2 + front 1, 640×360 | 29.9 / 29.8 | 83.66 / 85.24 | 84.49 / 86.46 |

The faster pair's median delivery was 79.11 / 82.20 ms. Capture-result delivery p99 was 85.43 / 87.55 ms. The previously measured Samsung preview path reached the texture at 105.3 ms p99 and remains a composite with unproven independent rear identity. The 50 ms end-to-end gate fails before preprocessing, and a qualified three-sensor route remains blocked.

The final APK repeated the 640×360 ultrawide and front trial. It matched 64 sampled image timestamps to capture results on each independently opened camera. Capture metadata reported 1.74 mm and 3.3 mm focal lengths, 90° and 270° sensor orientations, and median exposures of 19.98 and 33.24 ms. It delivered 246 ultrawide and 236 front frames at 29.89 and 29.79 fps. Acquisition p99 was 74.92 and 88.73 ms, and VisionIPC publication p99 was 76.04 and 90.25 ms, respectively. All 482 frames exceeded 50 ms. The observed 15.40 ms p99 skew is for ultrawide plus front.

The APK counts VisionIPC deliveries over 50 ms, records capture-result and image timestamp matches, and reports publication age separately from acquisition and packing. The physical diagnostic model path gates on the earlier rear timestamp or the current front timestamp, includes parsing and serialization in the deadline, and reports age rejections, frame gaps, dropped inputs, and late outputs. A send already in flight can complete late and counts as a miss. Diagnostic messages carry invalid status.

The separate [ARM64 emulator Camera2 run](../../build/three-camera-recovery.3N6tff/emulator-camera-timing.json) failed with 222 future capture timestamps in 224 virtual frames at 640×480. Measured publication-age p99 was −3.94 ms. The model gate rejects future capture times. The injected-input emulator tests exercised a separate path.

## Accelerator

The shell Vulkan diagnostic used original inputs, a hash-verified converted graph and weights, and extracted the complete graph's outputs as well as selected Vulkan tensors. Trace extraction retains intermediate buffers. Trace runs are diagnostic and excluded from timing and qualification.

The first mapped tolerance failure was `linear_122` (original `model/p_node_linear_5`), a 512×512 traffic projection. The captured input passed at the original `rtol=1e-3`, `atol=1e-4`. Selected earlier vision tensors, the road ReduceMax, and the mapped desire branch passed on seed 42. Marking the traffic projection for the existing ordered half kernel used an exact original weight/bias match.

| Seed 42 trace | Traffic projection values outside tolerance | Driving output values outside tolerance | Feature state values outside tolerance |
| --- | ---: | ---: | ---: |
| Unmodified Vulkan | 21 | 867 | 50 |
| Ordered traffic projection | 10 | 859 | 50 |
| Both traffic projections ordered | 0 | 848 | 50 |

All three runs had exact image and desire state. Ordering both projections made `linear_121`, its Swish output after the half cast, and `linear_122` exact for seed 42 in the trace. The [two-projection diagnostic](../../build/three-camera-recovery.3N6tff/road-ordered-both.json) ran against the final installed APK using a separately staged graph with SHA-256 `64dd4a8eb0bfb992f06fba2a9864cfcde7537db0cf763fbb25f605af9ddd542c`. It still failed full-graph output checks. Earlier MNN/OpenCL [795-comparison accuracy evidence](../full-model-controls-20261101/candidate45zc-delivery.md) belongs to its recorded APK. The [HTP primitive results](../npu-recovery-20261101/README.md) still block HTP promotion.

On the final APK, untapped ncnn Vulkan verified no CPU fallback for both models. Driver monitoring passed all three seeds, their sequences, and the 60-frame camera replay. On road, the output and feature-state mismatch counts were 867/50, 825/21, and 664/11 for seeds 42, 7, and 123. Image and desire history were exact. Road recurrence, full accelerator qualification, and formal accelerator timing stayed blocked at the seed gate.

## Controls and validation

Android controls now polls all subscribed inputs in one nonblocking native poll at the start of each absolute 10 ms slot. The selfdrive frequency anchor, legacy timing metrics, and deadline counters are preserved. The two 30-second Fold 3 runs below used the same [full-chain device replay fixture](../../build/three-camera-recovery.3N6tff/phone-candidate45-capture-budget-steady.json), SHA-256 `b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99`.

| Steady synthetic replay | Previous APK | Final APK |
| --- | ---: | ---: |
| Mean controls rate, Hz | 100.00 | 100.01 |
| Work p99, ms | 2.25 | 2.20 |
| Scheduled release to completion p99, ms | 4.21 | 4.18 |
| Complete cycle p99, ms | 12.39 | 12.45 |
| Raw cycle interval p99, ms | 11.22 | 11.16 |
| Input handling p99, ms | 1.58 | 1.50 |
| Input thread CPU p99, ms | 1.51 | 1.49 |
| Scheduler delay p99, ms | 1.31 | 1.24 |
| Reporting p99, ms | 0.35 | 0.37 |
| Complete cycle misses | 1238 | 1177 |
| Work misses / scheduled misses / skipped slots | 0 / 0 / 0 | 0 / 0 / 0 |

Controls work and scheduled completion passed in isolation. The 10 ms complete-cycle gate still failed. The separate Fold 3 fault fixture passed factory ACC cancellation and rejected stale CAN. Both fixture runs produced the required outputs, with zero unsafe outputs and no worker failures.

The Android 15 ARM64 emulator passed the steady and fault fixtures, including factory ACC cancellation and stale CAN rejection. The steady run measured 99.96 Hz, 2.36 ms work p99, 13.52 ms complete-cycle p99, 1,342 complete-cycle misses, one scheduled deadline miss, and one skipped slot. The [30-second MetaDrive run](../../build/three-camera-recovery.3N6tff/emulator-candidate45-capture-budget-metadrive.json) passed its synthetic steering feedback and stale-input tests. It processed 2,630 packets, with 1,628 active steering samples and 921 inactive samples after fault injection. Longitudinal actuation and below-minimum-speed steering each recorded zero active samples.

Focused host checks passed 215 tests, and 8 Android Java unit tests passed. The combined Hyundai behavior, Hyundai safety, Android simulator protocol, and MetaDrive bridge test command completed 1,921 tests with 202 skips. Both devices passed 28 instrumentation tests covering runtime, camera, and perception fixtures, thermal behavior, and USB rejection. Raylib reported 29.52 fps in the Fold 3 fault fixture and 29.50 fps on the emulator during MetaDrive. The [Fold 3 lifecycle check](../../build/three-camera-recovery.3N6tff/phone-lifecycle.json) passed background/resume and renderer recreation with the manager and worker process IDs intact. A physical fold transition was outside that trial.

The [emulator lifecycle check](../../build/three-camera-recovery.3N6tff/emulator-lifecycle.json) passed background/resume, display resizing from 1480×720 to 1280×720, restoration, and renderer recreation with the manager and worker process IDs intact.

Final cleanup on each device confirmed stopped app and test processes, released camera ownership, no ADB forwarding, and the same installed APK hash. The Fold 3 finished with 32.47 continuous seconds of NORMAL thermal status at 25.9°C skin temperature. The emulator finished with 31.14 seconds of NORMAL status and shut down.

| Acceptance gate | Result |
| --- | --- |
| APK identity on Fold 3 and ARM64 Android 15 emulator | Passed for both application and instrumentation APKs |
| Camera-only 50 ms gate | Failed before inference |
| Emulator virtual Camera2 timestamps | Failed |
| Raylib surface recovery | Passed on both devices, including display resizing on the emulator |
| Simultaneous rear wide, rear ultrawide and front with verified rear identity and geometry | Blocked |
| Driver model original-reference suite on Fold 3 Vulkan | Passed 159 comparisons without CPU fallback |
| Road model original-reference suite on Fold 3 Vulkan | Failed on all three seeds |
| 100 Hz controls work and scheduled completion on the Fold 3 fixture | Passed in isolation |
| 10 ms complete-cycle gate | Failed on both devices |
| 20 Hz per-model end-to-end, simultaneous cameras and GPU/controls/UI at load, extended runs | Blocked at preceding gates |
| Vehicle mounting, calibration, passive Panda and vehicle validation | Blocked by unavailable hardware and mount |

## Reproduction

Run from the repository root with the provisioned ARM64 NDK, SDK, patched accelerator libraries and staged model references. The [previous road delivery](../road-vulkan-20261105/README.md) records the pinned road graph staging commands.

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME="$HOME/Library/Android/sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python"
android/gradlew -p android --offline --console plain -Pfold3CameraCompat=true \
  -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest

export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
export CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
.venv/bin/python -m unittest discover -s android/tests
.venv/bin/python -m unittest opendbc.safety.tests.test_hyundai opendbc.car.hyundai.tests.test_hyundai \
  openpilot.tools.sim.tests.test_android_protocol openpilot.tools.sim.tests.test_metadrive_bridge

serial=RFCR81Y1TBJ
pkg=com.samsung.sunnypilot.fold3
run_dir="$(mktemp -d "$PWD/android/build/three-camera-repro.XXXXXX")"
candidate="$PWD/android/build/candidates/candidate45-capture-budget"
adb -s "$serial" install -r "$candidate/app-debug.apk"
adb -s "$serial" install -r "$candidate/app-debug-androidTest.apk"
.venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
  --adb "$ANDROID_HOME/platform-tools/adb" --serial "$serial" --candidate "$candidate" \
  --output "$run_dir/installed.json"
.venv/bin/python android/scripts/probe_ncnn.py --serial "$serial" --model driving_supercombo \
  --trace-count 36 --report "$run_dir/road-linear.json"
.venv/bin/python android/scripts/compare_ncnn_trace.py "$run_dir/road-linear.json"
.venv/bin/python android/scripts/probe_ncnn.py --serial "$serial" --model driving_supercombo \
  --trace-count 18 --ordered-half-node model/p_node_linear_5 --report "$run_dir/road-ordered.json"
.venv/bin/python android/scripts/compare_ncnn_trace.py "$run_dir/road-ordered.json"
.venv/bin/python android/scripts/probe_ncnn.py --serial "$serial" --model driving_supercombo \
  --trace-count 18 --ordered-half-node model/p_node_linear_4 --ordered-half-node model/p_node_linear_5 \
  --report "$run_dir/road-ordered-both.json"
.venv/bin/python android/scripts/compare_ncnn_trace.py "$run_dir/road-ordered-both.json"

.venv/bin/python android/scripts/experiment.py --serial "$serial" --package "$pkg" \
  --report "$run_dir/camera-isolation.json" -- .venv/bin/python android/scripts/validate_cameras.py \
  --serial "$serial" --package "$pkg" --ids 2,1 --width 640 --height 360 --seconds 8 \
  --output "$run_dir/camera.json"

.venv/bin/python android/scripts/experiment.py --serial "$serial" --package "$pkg" \
  --report "$run_dir/models-isolation.json" -- .venv/bin/python android/scripts/validate_models.py \
  --serial "$serial" --package "$pkg" --profile ncnn-vulkan --report "$run_dir/models.json"

.venv/bin/python android/scripts/experiment.py --serial "$serial" --package "$pkg" \
  --report "$run_dir/instrumentation-isolation.json" -- .venv/bin/python android/scripts/validate_instrumentation.py \
  --serial "$serial" --package "$pkg" --expected-tests 28 --output "$run_dir/instrumentation.txt" \
  --report "$run_dir/instrumentation.json"

.venv/bin/python android/scripts/experiment.py --serial "$serial" --package "$pkg" --runtime full-chain \
  --report "$run_dir/steady-isolation.json" -- .venv/bin/python -m openpilot.tools.sim.android_chain_validation \
  --serial "$serial" --package "$pkg" --seconds 30 --device-replay --steady-inputs --report "$run_dir/steady.json"
```

Repeat the last command without `--steady-inputs` under a distinct output name for factory ACC cancellation and stale CAN. The same APK pair and fixtures can run on the dedicated ARM64 API 35 emulator. Verify each installed app and test APK using [verify_installed.py](../full-model-controls-20261101/verify_installed.py) and the archived candidate manifest. Each wrapper stops the experiment and checks camera release and forwarding cleanup. Its pre-run cooldown requires 30 continuous seconds of NORMAL thermal status and, on the phone, skin temperature at or below 32°C.

```sh
.venv/bin/python android/scripts/experiment.py --serial "$serial" --package "$pkg" --runtime simulation \
  --report "$run_dir/phone-lifecycle-isolation.json" -- .venv/bin/python android/scripts/validate_lifecycle.py \
  --serial "$serial" --package "$pkg" --report "$run_dir/phone-lifecycle.json"
```

```sh
"$ANDROID_HOME/emulator/emulator" -avd sunnypilot_fold3_candidate45_api35 -port 5560 \
  -no-window -no-audio -no-boot-anim -no-snapshot -gpu swiftshader -feature -Vulkan \
  -memory 4096 -cores 4 -camera-back emulated -camera-front emulated
# In a separate terminal with the same variables, after boot:
serial=emulator-5560
adb -s "$serial" install -r "$candidate/app-debug.apk"
adb -s "$serial" install -r "$candidate/app-debug-androidTest.apk"
.venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
  --adb "$ANDROID_HOME/platform-tools/adb" --serial "$serial" --candidate "$candidate" \
  --output "$run_dir/emulator-installed.json"
.venv/bin/python android/scripts/experiment.py --serial "$serial" --package "$pkg" --runtime simulation \
  --report "$run_dir/metadrive-isolation.json" -- .venv/bin/python -m openpilot.tools.sim.android_validation \
  --serial "$serial" --metadrive --duration 30 --fault stale --report "$run_dir/metadrive.json"
.venv/bin/python android/scripts/experiment.py --serial "$serial" --package "$pkg" \
  --report "$run_dir/emulator-camera-isolation.json" -- .venv/bin/python android/scripts/validate_cameras.py \
  --serial "$serial" --package "$pkg" --ids 0 --width 640 --height 480 --seconds 8 \
  --output "$run_dir/emulator-camera.json"
.venv/bin/python android/scripts/experiment.py --serial "$serial" --package "$pkg" --runtime simulation \
  --report "$run_dir/emulator-lifecycle-isolation.json" -- .venv/bin/python android/scripts/validate_lifecycle.py \
  --serial "$serial" --package "$pkg" --resize --report "$run_dir/emulator-lifecycle.json"
adb -s "$serial" emu kill
```
