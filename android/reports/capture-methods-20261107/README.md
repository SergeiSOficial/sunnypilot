# Fold 3 capture methods, 7 November 2026

Candidate52 adds an opt-in constrained-high-speed Camera2 diagnostic and model-size GPU readback. Separate 30-second wide and front runs reach camera-to-publication maxima of **44.89 ms** and **38.91 ms**. Three-camera capture remains **blocked** by latency and independent rear-sensor evidence. These camera-only runs exclude model inference and cannot establish the requested 50 ms sensor-to-model-output limit. Vehicle TX remains compiled out and engagement disabled.

## Artifacts

The branch is `codex/fold3-android-port`, based on workspace HEAD `7631d5248cc9efd10b962107aadacbb908a9f14d`. Changes are local. Existing candidates and their failed receipts are retained.

| Artifact | SHA-256 |
| --- | --- |
| [Candidate50 app](../../build/candidates/candidate50-capture-trace/app-debug.apk) | `52bb80d8109743cc0b136187c80382f2ff9b2014778d27af1963056eedfaa192` |
| [Candidate51 app](../../build/candidates/candidate51-high-speed-camera/app-debug.apk) | `0e151f7002b5951eb41384435e64babdd0ed6b5e00894b50ccc1b07a7261de50` |
| [Final candidate52 app](../../build/candidates/candidate52-high-speed-model-size/app-debug.apk) | `3af70fdb3f8e68d9c9d720494c05c216ddedd7133d3640e0c2deb4be4e43f749` |
| [Final candidate52 instrumentation](../../build/candidates/candidate52-high-speed-model-size/app-debug-androidTest.apk) | `8c12ff89baa3eff616c2886eb8f27ebf634976a72cfcc60cb7170b921eba04b6` |

Final app hashes were verified after installation on SM-F926B `RFCR81Y1TBJ` and ARM64 Android 15 `emulator-5586`. Experiment wrappers also checked unchanged APK hashes before and after each run. Source snapshots accompany the final artifact. [Fold 3 installed hash](../../build/candidates/candidate52-high-speed-model-size/fold-installed-sha256.txt) · [Emulator installed hash](../../build/candidates/candidate52-high-speed-model-size/emulator-installed-sha256.txt) · [Source hashes](../../build/candidates/candidate52-high-speed-model-size/source-sha256.txt)

## Capture implementation

The new `high-speed-120` profile requires one public camera and one GPU-backed PRIVATE surface. It checks the advertised capture size and fixed 120 fps range, asks whether the high-speed session configuration is supported, and submits the platform-generated high-speed request list. Unsupported cameras and combinations are rejected before opening. The existing vendor-crash exclusions remain enforced. Session batching follows the platform API. [Android constrained-high-speed reference](https://developer.android.com/reference/android/hardware/camera2/CameraConstrainedHighSpeedCaptureSession)

Candidate51 reads back the full 1280×720 image. Candidate52 keeps that advertised capture size and uses the existing GLES conversion path to render the complete image to 960×540. Reports retain capture size, rendered size, orientation and `full_frame_bilinear` resampling. The instrumentation fixture checks scaled color and orientation. Buffers and PBO/fence queues remain bounded.

The fixed 120 fps request delivered approximately **59.4 fps wide** and **59.6 fps front** to the app. Reported throughput uses observed frames. This profile is diagnostic-only and retains `visionipc_valid=false`. High-speed results contain incomplete crop and focal metadata, so optical geometry and calibration remain unqualified.

Capture traces retain raw sensor time, elapsed-realtime/monotonic samples, frame number, exposure, frame duration, callback arrival, rolling-shutter skew and reported pipeline depth. The prior seven-column prefix is preserved. Callback events join only matching frames and exposure timestamps in a bounded ring. Missing fields remain `-1`. Neither device advertises readout-timestamp support in these trials. Invalid or future capture times remain rejected. [Android camera timestamp reference](https://developer.android.com/reference/android/hardware/camera2/CameraCharacteristics#SENSOR_INFO_TIMESTAMP_SOURCE)

## Isolated timing

All measurements below are camera-only, with one camera active per PRIVATE trial. Publication age begins at the reported sensor capture time. Stage percentiles are measured independently. The output-size change was selected as a reversible diagnostic improvement after candidate51 exceeded 50 ms once during its wide 30-second run.

| Candidate and route | Duration | Capture → rendered size | Observed fps | Acquisition p99 | Publication p99 | Publication max | Published over 50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 51 wide regular preview | 8 s | 1280×720 → 1280×720 | 29.88 | 94.47 ms | 99.45 ms | 100.05 ms | 237 |
| 51 wide high-speed | 30 s | 1280×720 → 1280×720 | 59.42 | 37.50 ms | 43.40 ms | **51.13 ms** | **1** |
| 52 wide high-speed | 30 s | 1280×720 → 960×540 | 59.42 | 37.40 ms | **41.84 ms** | **44.89 ms** | **0** |
| 51 front regular preview | 8 s | 1280×720 → 1280×720 | 29.91 | 83.90 ms | 88.90 ms | 89.68 ms | 237 |
| 51 front high-speed | 30 s | 1280×720 → 1280×720 | 59.61 | 26.14 ms | 32.54 ms | **35.96 ms** | 0 |
| 52 front high-speed | 30 s | 1280×720 → 960×540 | 59.61 | 26.02 ms | **30.97 ms** | **38.91 ms** | **0** |

The 30-second comparisons use the same capture size, requested mode and duration. Candidate52 lowers publication p99 by 1.56 ms wide and 1.57 ms front. Front maximum age increases by 2.95 ms despite the lower p99. Regular-preview runs have different frame rates and shorter durations, shown explicitly above. [Machine-readable comparison with all 16 physical trials](comparison.json)

Candidate52 wide acquires 1,779 frames and publishes 1,778. One acquired frame lacks publication at teardown and remains counted in the receipt. Front acquires and publishes all 1,785 frames. Both report zero invalid capture timestamps, lost buffers, dropped PBO readbacks, superseded publications and coalesced texture callbacks. The diagnostic camera-only gate passes for observed publications. The unpublished frame and absent inference prevent a zero-miss end-to-end qualification claim. [Wide 30-second receipt](../../build/candidates/candidate52-high-speed-model-size/fold-wide-high-speed-30s.json) · [Front 30-second receipt](../../build/candidates/candidate52-high-speed-model-size/fold-front-high-speed-30s.json)

High-speed capture shortens available exposure. Present result fields report approximately 8.33 ms exposure and 4.89/3.78 ms wide/front rolling-shutter skew. Only 32 of the final 64 sampled high-speed results contain exposure metadata. Bench images therefore provide timing evidence without establishing low-light suitability or calibrated road/driver input geometry.

## Simultaneous cameras

Public ultrawide camera `2` advertises no constrained-high-speed mode. Its high-speed request is rejected before opening. The new single-camera path therefore cannot be extended to the required wide, ultrawide and front combination using the currently advertised configuration. Earlier public physical-output and concurrent-session restrictions remain recorded in the preceding report. [Ultrawide rejection](../../build/candidates/candidate51-high-speed-camera/fold-ultra-high-speed-rejected.json) · [Earlier public-route evidence](../camera-paths-20261106/README.md)

Samsung logical `20`, mode `37`, still delivers preview, thumbnail and front streams together. Candidate50 compared preview, record and 720p-record profiles under the same 12-second continuous-readback workload. All three transport trials pass, while identity remains blocked.

| Candidate50 profile | Thumbnail delivery p99 | Rear preview delivery p99 | Front delivery p99 |
| --- | --- | --- | --- |
| Preview | 84.90 ms | 105.85 ms | 98.16 ms |
| Record | 83.82 ms | 103.69 ms | 97.55 ms |
| Record, 720p | 79.33 ms | 105.41 ms | 96.62 ms |

The final candidate52 record-profile repeat delivers 359 thumbnail frames, 355 rear-preview readbacks and 359 front frames in 12 seconds. Thumbnail, preview and front delivery p99 are **91.67/115.79/98.42 ms**. Preview acquisition p99 is **110.99 ms**, readback p99 **6.54 ms**, and sink arrival p99 **115.81 ms**. Capture delay exceeds the entire 50 ms budget before inference. This repeat is retained separately from candidate50 comparisons. [Final triple-stream receipt](../../build/candidates/candidate52-high-speed-model-size/fold-samsung-record-final.json)

The thumbnail remains sensor-unverified and may contain composite regions. Its parent logical timestamps, changing pixels and simultaneous callbacks cannot prove separate rear sensors. VisionIPC qualification remains false. Manual optical trials were deferred, and the phone remains unmounted. Independent rear identity, exact crop geometry, rear pairing within 20 ms and mounted calibration stay **blocked**.

Read-only inspection of the installed Samsung device feature metadata identifies Multi Recording and Slow Motion. Generic camera application code includes Dual Recording V2 mode `45`, but this device does not advertise it. That mode was excluded from experiments. Known mode `32` and private physical-output provider-crash configurations remain excluded. Only extracted capability facts and hashes are retained. [Stock camera capability facts](../../build/candidates/candidate50-capture-trace/stock-camera-capabilities.json)

## Validation

| Check | Result |
| --- | --- |
| Final ARM64 app, instrumentation APK and Java unit tests | **Passed**, nine Java tests. [Build](../../build/candidates/candidate52-high-speed-model-size/build-final.log) · [Unit results](../../build/candidates/candidate52-high-speed-model-size/java-unit-tests.json) |
| Host preprocessing, clock, IPC, scheduling, conversion and safety checks | **Passed**, 220 discovered, three skipped because OpenCV is unavailable. Optical qualification remains open. [Host log](../../build/candidates/candidate52-high-speed-model-size/host-tests.txt) |
| Final Fold 3 instrumentation | **Passed**, 28 tests, including GPU conversion/scaling, lifecycle, controls metrics, supervision, perception rejection and native USB/TX restrictions. [Receipt](../../build/candidates/candidate52-high-speed-model-size/fold-instrumentation.json) |
| Final Android 15 ARM64 emulator instrumentation | **Passed**, the same 28 tests. [Receipt](../../build/candidates/candidate52-high-speed-model-size/emulator-instrumentation.json) |
| Unsupported high-speed mode rejection | **Passed negative test**, ultrawide and emulator requests remain blocked without opening cameras. [Fold 3](../../build/candidates/candidate51-high-speed-camera/fold-ultra-high-speed-rejected.json) · [Emulator](../../build/candidates/candidate52-high-speed-model-size/emulator-high-speed-rejected.json) |
| Actual emulator camera timing | **Failed**, invalid/future timestamps and stale-input termination. Ten images publish, 71 timestamps are rejected, and publication p99 reaches 166.65 ms. Arrival-time substitution remains disabled. [Receipt](../../build/candidates/candidate52-high-speed-model-size/emulator-camera.json) |
| Separate wide/front short and 30-second camera-only timing | **Passed observed-publication gate** on candidate52, with the teardown count above retained |
| Three independent calibrated cameras within 50 ms | **Blocked**, rear identity and geometry missing, measured simultaneous route exceeds budget |
| Full 795 model comparisons, combined controls/GPU workload and MetaDrive | **Not rerun in this capture experiment**. Earlier accuracy and controls failures remain open |
| Two-minute and 30-minute combined acceptance | **Blocked** by preceding camera, model and controls gates |
| Physical Panda, firmware, harness and vehicle behavior | **Blocked**, Panda disconnected and phone unmounted |

The candidate50 first build caught an outdated seven-column test expectation after trace expansion. It was updated to twelve columns and rerun successfully. An initial host run used an incorrect macOS SDK and failed native links. The corrected SDK run passed. Candidate51's first final Samsung retry omitted required `--logical 20` and was rejected by the CLI before opening cameras. Candidate52 reran it with the required argument. Earlier failures remain available alongside their corrected runs. A full-working-tree static quality scan was stopped after it failed to return within the bounded review window, so it supplies no additional acceptance evidence. [Initial build log](../../build/candidates/candidate50-capture-trace/build-initial.log) · [Initial host log](../../build/candidates/candidate50-capture-trace/host-tests.txt) · [Corrected host log](../../build/candidates/candidate50-capture-trace/host-tests-sdk-fixed.txt) · [Rejected retry](../../build/candidates/candidate51-high-speed-camera/fold-samsung-record-final.log)

## Cleanup

Each instrumented experiment retained hash checks, clean app/test exits, camera release, unchanged provider PIDs, forwarding cleanup and at least 30 continuous seconds of NORMAL thermal status at skin temperature at or below 32°C. Validation failures are separate from isolation results. The final triple-stream wrapper records failed validation and passed isolation.

Final physical cleanup confirms zero app/test processes, no owned camera and empty forward/reverse mappings, with 32.25 continuous seconds NORMAL at 29.5°C. Emulator cleanup confirms the same with 30.82 seconds NORMAL at 30.8°C, followed by emulator shutdown. Cooling changes remain deferred and thermal protection remains active. [Fold 3 cleanup](../../build/candidates/candidate52-high-speed-model-size/fold-final-cleanup.json) · [Emulator cleanup](../../build/candidates/candidate52-high-speed-model-size/emulator-final-cleanup.json)

## Reproduce

Use the provisioned Android SDK/NDK, staged models and patched engine builds. Run from the repository root.

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME="$HOME/Library/Android/sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python"
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
android/gradlew -p android --offline --console plain -Pfold3CameraCompat=true \
  -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
.venv/bin/python -m unittest discover -s android/tests

p=com.samsung.sunnypilot.fold3
s=RFCR81Y1TBJ
out=$(mktemp -d "$PWD/android/build/capture-methods.XXXXXX")
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
    --output "$out/$name.json" "$@"
}
trial wide --ids 0 --width 1280 --height 720 --seconds 30 \
  --camera-profile high-speed-120 --private-gpu
trial front --ids 1 --width 1280 --height 720 --seconds 30 \
  --camera-profile high-speed-120 --private-gpu
trial samsung --ids 0,2,1 --logical 20 --samsung-multiview --continuous-readback \
  --width 960 --height 540 --seconds 12 --camera-profile record
```

The Samsung trial returns a failing wrapper exit for blocked qualification. Inspect `validation_status` and `isolation_status` separately. The high-speed profile rejects multi-camera requests by design. Full-model and vehicle acceptance remain gated on their separately recorded evidence.
