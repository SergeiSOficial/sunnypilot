# Fold 3 simultaneous-camera paths, 6 November 2026

Branch `codex/fold3-android-port`. The camera harness exercises direct Camera2 YUV, Camera2 PRIVATE surfaces with GLES NV12 conversion, and Samsung mode 37. All trials use the compatibility package. Camera output remains diagnostic. Vehicle TX is compiled out and physical engagement is disabled.

| Artifact | Application SHA-256 | Evidence |
| --- | --- | --- |
| [Candidate49 APK](../../build/candidates/candidate49-camera-paths/app-debug.apk) | `28f4bed84c881df71b627f5a271941a1ba7d1d53bf5b9825e93701a40de6e205` | [Final guard and retests](../../build/camera-paths-candidate49.Q4ii2k/) |
| [Candidate48 APK](../../build/candidates/candidate48-camera-paths/app-debug.apk) | `1c0e69243653f483b40ce107a6e26c33f49439603330a3ca2e86728221edba4b` | [PRIVATE trials and vendor crash](../../build/camera-paths-candidate48.rLmqNU/) |
| [Candidate47 APK](../../build/candidates/candidate47-camera-paths/app-debug.apk) | `81f7c28e8928421718a0e3c5b6fb32e7e153121ee62ee0be8d8fab3d1527dbe4` | [YUV comparisons and optical trials](../../build/camera-paths-final.VPGOvx/) |
| Candidate46 | `f21cd513c0952f8b6805865cb2b786b5dd87aa70b3797560a7a78a45196c7f12` | [Initial public probes](../../build/camera-paths.KJtEXs/) |

The [instrumentation APK](../../build/candidates/candidate49-camera-paths/app-debug-androidTest.apk) SHA-256 is `96876c828001aafe773f8d00589528522dce16709372ab77e134117186de6a6f`. Both APK hashes matched on the Fold 3 and Android 15 ARM64 emulator. [Fold 3 install](../../build/camera-paths-candidate49.Q4ii2k/fold-installed-sha256.txt) · [Emulator install](../../build/camera-paths-candidate49.Q4ii2k/emulator-installed-sha256.txt)

Candidate47's PRIVATE sessions hit a null ImageReader when binding request targets. Candidate48 binds each request to the same surface used to configure its session. Candidate49 excludes the wide/front PRIVATE combination on SM-F926B after the stock provider crashed when tearing down that configuration on candidate48. The guard opened zero cameras and the provider PID stayed stable. [Vendor crash excerpt](../../build/camera-paths-candidate48.rLmqNU/gpu-wide-provider-crash.log) · [Guard receipt](../../build/camera-paths-candidate49.Q4ii2k/gpu-wide-blocked-experiment.json)

## Camera support and identity

- **Blocked** for public logical physical outputs. The Fold 3 lists public camera IDs 0, 1, 2 and 3, with no physical group on any of them. Samsung IDs 20 and 23 return characteristics for rear groups outside the public camera list. The public path stops before opening those IDs. The recorded provider-crash exclusions still apply. [Camera2 multi-camera guide](https://developer.android.com/media/camera/camera2/multi-camera) · [Camera history](../../CAMERAS.md)
- **Blocked** for three independently opened public devices. Camera2 advertises an empty set of concurrent combinations on this device. Both established rear-plus-front pairs remain available as diagnostic controls. The logical path checks the advertised combination, the concurrent output configuration, and support for physical outputs within the logical session before creating the session. [CameraManager concurrency API](https://developer.android.com/reference/android/hardware/camera2/CameraManager#getConcurrentCameraIds())
- **Passed** for Samsung mode 37 transport. Preview, thumbnail, and front ran together in three stationary bench trials. Each rear clip contributed 25 pairs within 20 ms. The provider returned no physical result IDs. The two rear vendor outputs shared a capture timestamp, so measured stream overlap cannot establish independent physical exposure timing.
- **Blocked** for independently qualified Samsung rear feeds. Across three samples in each trial, SIFT and homography comparison found 222–245 ultrawide inliers for the thumbnail and 17–28 wide inliers. The preview required a quarter-turn and rescaling to restore a landscape aspect. The remapped preview had 39–54 wide inliers across 10–12 of 16 grid cells and 13–18 ultrawide inliers. The direct wide and ultrawide references had 19 mutual inliers. These results indicate different lens views in the stationary scene, but they leave vendor crop and any mixed regions unqualified. [Optical comparison](../../build/camera-paths-final.VPGOvx/optics.json)

The bench used available ambient light and an unmounted phone. Capture receipts retain exposure duration, crop, focal length, orientation, and clock samples where available. The GLES orientation fixture passes. Direct-YUV versus PRIVATE geometric equivalence on physical camera pixels, manual lens occlusion, independent physical rear capture timing, and mounted calibration remain open.

## Fold 3 camera timing

Each pair ran for six seconds with a target of 30 fps and delivered about 29.7–29.9 fps per camera. The table orders each pair rear then front. Acquisition and publication ages start at sensor capture. GPU conversion ran on GLES3 and used the existing bounded PBO/fence path. All six failed the 50 ms camera-only gate. The wide/front PRIVATE timing is diagnostic because the vendor provider crashed at teardown.

| APK | Path and size | Capture to acquisition p99 | Capture to VisionIPC p99 | Samples over 50 ms |
| --- | --- | ---: | ---: | ---: |
| 47 | Ultrawide/front YUV record, 640×360 | 77.60 / 92.39 ms | 78.22 / 93.07 ms | 185 / 176 |
| 47 | Ultrawide/front YUV preview, 640×360 | 79.66 / 91.39 ms | 80.12 / 91.99 ms | 186 / 176 |
| 48 | Ultrawide/front PRIVATE preview, 640×360 | 77.83 / 88.75 ms | 80.44 / 92.27 ms | 186 / 178 |
| 47 | Wide/front YUV record, 640×480 | 85.24 / 90.51 ms | 86.06 / 90.93 ms | 185 / 176 |
| 47 | Wide/front YUV preview, 640×480 | 77.79 / 89.17 ms | 79.27 / 90.14 ms | 185 / 176 |
| 48 | Wide/front PRIVATE preview, 640×480, provider crash | 88.93 / 91.78 ms | 91.79 / 94.27 ms | 186 / 178 |
| 49 | Ultrawide/front PRIVATE preview retest, 640×360 | 80.17 / 87.49 ms | 82.44 / 90.04 ms | 186 / 179 |
| 49 | Wide/front YUV preview, 640×360 | 74.99 / 82.66 ms | 75.84 / 83.36 ms | 185 / 177 |

Samples over budget include in-flight GPU callbacks at teardown. The receipts record successful VisionIPC sends, raw acquisitions, rejected timestamps and lost buffers separately. GPU packing took 3.11–4.75 ms p99. The GPU path still spent the full 50 ms budget waiting for acquisition. Samsung mode 37 preview acquisition ran at 103.43–124.15 ms p99 and delivery at 107.30–127.43 ms p99 across the three optical trials. The final retest without clip export delivered 170 preview, 178 thumbnail, and 177 front frames. Preview acquisition was 88.65 ms p99 and preview delivery was 102.19 ms p99. That run used a different workload from the optical trials.

The initial candidate46 ultrawide/front YUV record run delivered 73.86 / 88.45 ms p99, with 186 / 176 over-budget samples. The later matched results vary with capture conditions and give no evidence that GLES conversion or changing the request template closes the timing gap. [Initial run](../../build/camera-paths.KJtEXs/yuv-ultrawide-record.json)

**Overall three-camera qualification blocked.** The 50 ms sensor-to-model-output budget fails at camera acquisition, and no stock route with independent rear identity and geometry qualified. Combined 30-second, two-minute, and 30-minute runs were gated on camera timing. Accelerator model parity and controls complete-cycle timing remain tracked separately in the [three-camera recovery report](../three-camera-recovery-20261106/README.md). Bench evidence leaves mounted calibration and physical Panda and vehicle validation open.

## Validation

| Check | Result |
| --- | --- |
| ARM64 APK build and eight Java unit tests | Passed |
| Focused host suite | Passed, 219 tests including frame evidence and thermal checks. [Host log](../../build/camera-paths-final.VPGOvx/host-tests.txt) |
| Hyundai behavior and safety, Android simulator protocol, MetaDrive bridge | Passed, 1,921 tests run with 202 skips. [Host log](../../build/camera-paths-final.VPGOvx/vehicle-sim-host-sdk26.txt) |
| Final APK instrumentation | Passed, 28 tests on each device. [Fold 3](../../build/camera-paths-candidate49.Q4ii2k/fold-instrumentation.json) · [Emulator](../../build/camera-paths-candidate49.Q4ii2k/emulator-instrumentation.json) |
| Emulator Camera2 clock with PRIVATE surface | Failed. GLES3 converted 70 virtual frames, but the stream generated future timestamps and stopped on the freshness watchdog. Six frames reached VisionIPC. [Emulator camera receipt](../../build/camera-paths-candidate49.Q4ii2k/emulator-camera.json) |
| Emulator Raylib lifecycle | Passed background/resume, resized surface, restored surface, and renderer recreation with the manager and workers preserved. The first invocation lacked a freshly started runtime and timed out. [Initial receipt](../../build/camera-paths-candidate49.Q4ii2k/emulator-lifecycle.json) · [Simulation retry](../../build/camera-paths-candidate49.Q4ii2k/emulator-lifecycle-retry.json) |
| Fold 3 Raylib lifecycle | Passed background/resume and renderer recreation with the manager and workers preserved. A physical fold transition remains untested. The first invocation lacked a freshly started runtime and timed out. [Initial receipt](../../build/camera-paths-candidate49.Q4ii2k/fold-lifecycle.json) · [Simulation retry](../../build/camera-paths-candidate49.Q4ii2k/fold-lifecycle-retry.json) |

The candidate49 runs passed experiment isolation. Final cleanup confirmed zero app or test processes, zero app camera clients, and empty ADB forward and reverse lists on both devices. The Fold 3 ended with 32.25 continuous seconds of NORMAL thermal status at 27.3°C skin temperature. The emulator ended with 30.94 seconds of NORMAL status and was shut down after cleanup. [Fold 3 cleanup](../../build/camera-paths-candidate49.Q4ii2k/fold-retry-final-cleanup.json) · [Emulator cleanup](../../build/camera-paths-candidate49.Q4ii2k/emulator-retry-final-cleanup.json)

## Reproduce

Use the provisioned ARM64 NDK, SDK and staged model assets. Run from the repository root.

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME="$HOME/Library/Android/sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python"
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
android/gradlew -p android --offline --console plain -Pfold3CameraCompat=true \
  -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest

p=com.samsung.sunnypilot.fold3
s=$(adb -d get-serialno)  # For the emulator, set s to its emulator-XXXX ID.
out=$(mktemp -d "$PWD/android/build/camera-paths.XXXXXX")
adb -s "$s" install -r -g android/app/build/outputs/apk/debug/app-debug.apk
adb -s "$s" install -r -g android/app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
for name in "$p" "$p.test"; do
  installed=$(adb -s "$s" shell pm path "$name")
  adb -s "$s" shell sha256sum "${installed#package:}"
done

trial() {
  name=$1; shift
  .venv/bin/python android/scripts/experiment.py --serial "$s" --package "$p" \
    --report "$out/$name-experiment.json" -- \
    .venv/bin/python android/scripts/validate_cameras.py --serial "$s" --package "$p" \
    --seconds 6 --width 640 --height 360 --output "$out/$name.json" "$@"
}

trial public-triple --ids 0,2,1
trial public-logical --ids 0,2,1 --logical 20
trial wide-reference --ids 0 --height 480 --camera-profile preview --evidence-directory "$out/wide-reference"
trial ultrawide-reference --ids 2 --camera-profile preview --evidence-directory "$out/ultrawide-reference"
for i in 1 2 3; do
  trial "samsung-$i" --ids 0,2,1 --logical 20 --samsung-multiview --continuous-readback \
    --evidence-directory "$out/samsung-$i"
done
trial yuv-ultrawide-record --ids 2,1
trial yuv-ultrawide-preview --ids 2,1 --camera-profile preview
trial gpu-ultrawide-preview --ids 2,1 --camera-profile preview --private-gpu
trial yuv-wide-record --ids 0,1 --height 480
trial yuv-wide-preview --ids 0,1 --height 480 --camera-profile preview
trial yuv-wide-360-preview --ids 0,1 --camera-profile preview
trial gpu-wide-preview --ids 0,1 --height 480 --camera-profile preview --private-gpu
```

On the final build `gpu-wide-preview` must return blocked without opening any camera on the Fold 3.

The experiment wrapper returns nonzero when a camera check fails or is blocked. Read the camera receipt for that status and the experiment receipt for isolation. Stop the series if isolation fails or a camera provider restarts. Keep the Fold 3 stationary during the optical trials. Camera evidence is written only when requested and is removed from the phone after export. A separate host environment with NumPy and OpenCV runs the optical diagnostic and its tests.

```sh
uv run --no-project --with opencv-python-headless==5.0.0.93 --with numpy \
  android/scripts/compare_camera_optics.py "$out" "$out/optics.json"
.venv/bin/python android/scripts/experiment.py --serial "$s" --package "$p" \
  --report "$out/instrumentation-experiment.json" -- \
  .venv/bin/python android/scripts/validate_instrumentation.py --serial "$s" --package "$p" \
  --expected-tests 28 --report "$out/instrumentation.json" --output "$out/instrumentation.txt"
.venv/bin/python android/scripts/experiment.py --serial "$s" --package "$p" --runtime simulation \
  --report "$out/lifecycle-experiment.json" -- \
  .venv/bin/python android/scripts/validate_lifecycle.py --serial "$s" --package "$p" \
  --report "$out/lifecycle.json"

# Add --resize to the emulator lifecycle check. Use its emulated rear camera for the clock check.
trial emulator-camera --ids 0 --height 480 --camera-profile preview --private-gpu

# Use the provisioned repository venv with OpenCV available on its Python path.
.venv/bin/python -m unittest discover -s android/tests
.venv/bin/python -m unittest opendbc.safety.tests.test_hyundai opendbc.car.hyundai.tests.test_hyundai \
  openpilot.tools.sim.tests.test_android_protocol openpilot.tools.sim.tests.test_metadrive_bridge
```
