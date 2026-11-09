# Fold 3 accelerator, controls and camera repair, 7 November 2026

Candidate54 packages the road Vulkan spatial-mean repair, reusable controls-report buffers and an opt-in manual-exposure camera diagnostic. **The complete Android driving stack remains unqualified.** Road-model accuracy, the legacy controls complete-cycle gate and the three-camera 50 ms capture-to-model-output budget still fail or lack required evidence. Vehicle TX remains compiled out; physical engagement is disabled. Cooling optimization is deferred and thermal safeguards remain active.

## Artifacts and implementation

Branch `codex/fold3-android-port`, workspace HEAD `7631d5248cc9efd10b962107aadacbb908a9f14d`. These are local changes; earlier candidates, measurements and failed receipts are preserved.

| Artifact | SHA-256 |
| --- | --- |
| [Candidate53 app](../../build/candidates/candidate53-controls-buffers-manual-exposure/app-debug.apk) | `f5e9df953b902e75050a6dd1dd5819ca8ad9ea3c2529e78ba4cd7af8122778a5` |
| [Final candidate54 app](../../build/candidates/candidate54-road-mean/app-debug.apk) | `e4e6902dc2955005622b5b9358e4aa507afdcea99f74b904ada515c5b2a17db9` |
| [Final candidate54 instrumentation](../../build/candidates/candidate54-road-mean/app-debug-androidTest.apk) | `d87dff97acf3bef918dc66267cf7d59eac97e97c41dbed28df3114daa1d7d0ee` |
| Candidate54 packaged ARM64 ncnn library | `a8e3a7361b008b87433548938bc19af1bb23c70cb973d8fd796d56da828a0941` |

The app and test APK hashes match installation on SM-F926B `RFCR81Y1TBJ` and the dedicated ARM64 Android 15 `emulator-5586`. Each experiment also checks a stable installed app hash before and after execution. [Artifact and source identities](../../build/candidates/candidate54-road-mean/artifacts.json) · [Fold installed verification](../../build/candidates/candidate54-road-mean/fold-installed.json) · [Emulator installed verification](../../build/candidates/candidate54-road-mean/emulator-installed.json)

The ncnn patch is reproducible against source `7e36d25cb153092ba87a60530455c7376fedf452`, NDK `28.2.13676358`, API 35. It adds the exact road `[1,1024,4,8]` FP32 spatial-mean shape to the existing verified driver mean implementation, preserving eight-lane accumulation and horizontal reduction order. Existing driver arithmetic is unchanged. Original models, goldens, tolerances (`rtol=1e-3`, `atol=1e-4`) and the 795-comparison requirement are unchanged.

Controls timing samples now write into three preallocated owned double buffers. The bounded writer handoff swaps buffers and encodes only used samples; the manager process retains percentile calculation. No batch is overwritten to hide congestion. Consumer-stall, overflow, reporting-failure, heartbeat, cancellation, stale-input and watchdog behavior remain enforced. Absolute 10 ms start-of-slot pacing and existing deadline definitions are unchanged.

Trace comparison now matches exact weight, bias, shape and original-node provenance; duplicate heads require structurally identical provenance. It handles rank-four MemoryData and LayerNorm serialization correctly and separately hashes road precision graphs. Local captured-input operator checks remain diagnostic and cannot qualify full-model execution.

## Numerical evidence

The old road mean matches 600 of 1,024 captured values exactly, with maximum error `1.49e-8`. The repaired on-device shader matches **1,024/1,024 exactly**. Two targeted linear sections match **2,048/2,048** and **512/512** captured values exactly. No CPU fallback or unexpected non-finite outputs were observed in that diagnostic.

Nevertheless, the same seed-42 full-output road diagnostic still has **592 output elements and 29 feature-history elements outside original tolerances**; image and desire state agree. Its captured upstream inputs still differ, so local mean/linear agreement is insufficient. Diagnostic ordered-half graph changes are separately hashed and are not promoted into live inference. The standalone probe ran with candidate53 installed and the new native library subsequently packaged in candidate54; this is not a full-model candidate54 qualification result. [Probe receipt](../../build/road-mean-fixed-c53-seed42.json) · [Original-node trace comparison](../../build/road-mean-fixed-c53-seed42-comparison.json) · [Same-input comparisons](../../build/road-mean-fixed-c53-seed42-same-input.json)

**Candidate54 packaged untapped graphs:** driver monitoring passes **159/159** original-reference comparisons exactly: three initial seeds, 32-frame recurrence per seed and the pinned 60-frame replay. Road initial-output parity fails on all three seeds, so its recurrent/replay and all accelerator timing stages are not advanced. Both actual assignment audits verify no CPU fallback; all checked outputs are finite. No model is promoted for live perception. [Full model receipt](../../build/candidates/candidate54-road-mean/fold-models.json) · [Isolation and installed identity](../../build/candidates/candidate54-road-mean/fold-models-experiment.json)

| Candidate54 road seed | Output elements outside tolerance | Feature-history elements outside tolerance | Output max absolute error |
| --- | ---: | ---: | ---: |
| 42 | 867 | 50 | 0.162676 |
| 7 | 825 | 21 | 0.224926 |
| 123 | 664 | 11 | 0.073627 |

These numbers use the packaged graph; the 592/29 diagnostic result above uses additional separately hashed ordered-half annotations. Those are distinct artifacts, not a measured regression or improvement on a matched full graph.

Timing qualification remains blocked until all applicable initial, recurrent and pinned 60-frame replay comparisons pass without CPU fallback. Earlier complete strict-OpenCL/MNN numerical passes retain their original APK identities and measured performance failures; they do not qualify current Vulkan execution. QNN HTP remains diagnostic because its primitive accuracy prerequisites have not passed.

## Matched controls timing

These are separate cooled, 30-second on-device **synthetic-source process-chain fixtures**, not perception or physical-vehicle qualification. The fixture identity is `b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99`. Percentiles are not additive.

| Fold 3 fixture | Work p99, ms | Scheduled completion p99, ms | Legacy complete-cycle p99, ms | Reporting p99, ms | Legacy misses | Scheduled misses / skipped slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier matched APK `b3535754…` | 2.20 | 4.18 | 12.45 | 0.37 | 1177 | 0 / 0 |
| Candidate53 `f5e9df95…` | 2.289 | 4.125 | 12.392 | 0.392 | 1301 | 0 / 0 |
| Candidate54 `e4e6902d…` | 2.179 | 4.165 | 12.366 | 0.382 | 1213 | 0 / 0 |

Candidate54 averages **99.995 Hz** over 2,709 reported samples. Input handling p99 is 1.582 ms, scheduler delay 1.309 ms and active off-CPU time 0.223 ms. Work and scheduled-completion counters have zero misses in that fixture, but the retained legacy complete-cycle timing gate is **FAILED**. The buffer change does not establish a material end-to-end timing improvement. Raw intervals, pacing wait and every observed deadline counter remain visible. Metrics cover delivered batches; force-stop can leave an unflushed tail, so these counts are not asserted to cover every cycle of the 30-second interval. [Candidate53 fixture](../../build/candidates/candidate53-controls-buffers-manual-exposure/fold-controls-steady.json) · [Candidate54 fixture](../../build/candidates/candidate54-road-mean/fold-controls-steady.json) · [Earlier matched fixture](../three-camera-recovery-20261106/README.md)

The complete-cycle metric includes pacing since the preceding loop finished, then current input, work and reporting. Start-of-slot scheduling has no blocking input poll; variations in wakeup and active processing remain in the legacy measurement. It is not replaced by the smaller scheduled-completion measurement. Emulator timing is retained as emulator data and cannot establish Fold 3 performance.

## Camera experiments

Candidate53 adds `manual-8ms`, restricted to an independently opened public rear camera with advertised MANUAL_SENSOR support and compatible exposure, ISO and frame-duration limits. It requests an 8.333333 ms exposure and 33.333333 ms frame duration, recording both requests and applied capture-result settings. Front, logical and vendor paths are rejected before opening. This diagnostic does not qualify image quality or mounted geometry.

Separate eight-second direct-YUV trials at 640×360 measured about 29.88 fps. The phone was stationary on the bench, with no controlled illumination/quality comparison. All figures below exclude inference.

| Candidate53 camera/profile | Applied exposure, ms | Acquisition p99, ms | Capture-to-delivery p99, ms | 50 ms camera-only gate |
| --- | ---: | ---: | ---: | --- |
| Rear wide, record | 19.983 | 82.257 | 83.076 | FAILED |
| Rear wide, manual-8ms | 8.333 | 70.155 | 71.466 | FAILED |
| Rear ultrawide, record | 19.983 | 72.797 | 73.569 | FAILED |
| Rear ultrawide, manual-8ms | 8.333 | 60.552 | 62.314 | FAILED |

Applied manual ISO was 100, versus automatic ISO 178 wide / 574 ultrawide. All timestamp checks passed. Lower latency at the cost of different exposure is diagnostic evidence, not an accepted image-quality optimization. Independent rear `0+2` manual capture returned `camera_error:2:2` before delivering frames; the camera provider stayed alive and cleanup passed. No excluded provider-crash route was retried. [Exact camera commands](../../build/candidates/candidate53-controls-buffers-manual-exposure/fold-camera-commands.json)

The supported Samsung mode-37 transport delivered preview, thumbnail and front concurrently on candidate53 (709/358/358 frames). Capture-to-delivery p99 was **105.29 ms preview**, **88.65 ms thumbnail**, **97.41 ms front**. Independent rear sensor identity remains unproven; timestamp overlap and different pixels do not establish identity. The 50 ms sensor-to-model-output requirement therefore fails before inference. Public physical IDs and advertised concurrent sets remain absent. Earlier isolated high-speed wide/front camera-only results retain their own candidate52 identity; that profile does not support the required three-camera combination. [Prior capture methods](../capture-methods-20261107/README.md)

The **candidate54 final triple-stream rerun** again passes transport but returns blocked qualification: 710 preview, 359 thumbnail and 359 front frames, with zero timestamp/packing errors. Preview capture-to-sink p99 is **107.506 ms**, thumbnail delivery p99 **83.572 ms**, and front delivery p99 **96.103 ms**. Preview readback p99 is 6.628 ms versus texture acquisition p99 103.561 ms; acquisition remains the dominant measured stage. The provider remains unchanged and teardown passes. This is the final installed APK result, separate from candidate53 exposure trials. [Camera receipt](../../build/candidates/candidate54-road-mean/fold-three-camera.json) · [Experiment receipt](../../build/candidates/candidate54-road-mean/fold-three-camera-experiment.json)

The optical-analysis dependency was restored and all three optical-analysis tests now execute. Existing three-trial recordings still return **BLOCKED**, not independent rear proof. Lens-cover trials remain deferred. Mounted calibration, camera geometry in the final mount, physical Panda firmware/fingerprint/harness and vehicle behavior remain **BLOCKED** because the phone is unmounted and Panda is disconnected.

## Validation and experiment isolation

- **PASSED:** 224 host Android-port tests, zero skips, including preprocessing, timing/pairing, tracing, conversion, IPC, scheduling, reporting, supervision and receive-only restrictions. Final run: 27.542 seconds. Nine Java unit tests pass.
- **PASSED:** 1,921 focused Hyundai/Panda/simulator tests, with 202 expected skips; 4.218 seconds. Skipped cases are not treated as exercised physical tests.
- **PASSED:** candidate53 and candidate54 each complete all 29 instrumentation tests on both devices, including numerical-fault rejection, reporting integrity, permission/USB rejection, lifecycle and GPU conversion/orientation checks. Instrumentation micro-probes cannot establish full-model acceptance.
- Candidate53 passes steady and fault process-chain fixtures and lifecycle recovery on both devices, plus emulator display resize. Its MetaDrive stale-input test passes with 1,361 packets, 756 active and 426 inactive-after-fault outputs, no longitudinal commands or below-minimum-speed activation. This is simulator behavior, not road qualification.
- **FAILED:** native emulator camera timing contains future sensor times/coalesced frames; no arrival timestamps are substituted. Deterministic injected fixtures are reported separately.

**Candidate54 final functional reruns:** both devices pass steady-input and factory-ACC cancellation/stale-CAN process-chain fixtures with zero unsafe outputs and no worker failures. Both pass lifecycle recovery; emulator additionally passes display resize/recreation. The steady Fold fixture renders Raylib at 29.68 fps; lifecycle interruptions are not used as sustained UI benchmarks. Emulator MetaDrive passes the stale-input/cancellation fixture with 1,633 packets, 896 active and 479 inactive-after-fault outputs, zero longitudinal activation and zero below-minimum-speed activation. Maximum lateral error is 1.607 m in the explicitly synthetic damped steering plant; it is not a Hyundai vehicle or camera-perception test. [Fold fault fixture](../../build/candidates/candidate54-road-mean/fold-controls-faults.json) · [Emulator fault fixture](../../build/candidates/candidate54-road-mean/emulator-controls-faults.json) · [Fold lifecycle](../../build/candidates/candidate54-road-mean/fold-lifecycle.json) · [Emulator lifecycle](../../build/candidates/candidate54-road-mean/emulator-lifecycle.json) · [MetaDrive](../../build/candidates/candidate54-road-mean/emulator-metadrive.json)

Candidate54 emulator controls timing fails, as expected for this software-emulated workload: 96.59 Hz mean, work p99 14.494 ms, legacy complete-cycle p99 29.291 ms, 60 work misses, 170 scheduled misses, 1,326 legacy cycle misses and 95 skipped slots. This is not substituted for Fold 3 timing. The initial emulator camera request at 640×360 is rejected before opening because rear camera 0 does not advertise that size; a separate 640×480 run tests the supported stream configuration. [Unsupported-size rejection](../../build/candidates/candidate54-road-mean/emulator-camera.json)

The supported-size candidate54 emulator rerun delivers 229 rear and 227 front frames, but rejects every capture timestamp as future/invalid. Acquisition p99 is -18.35 ms rear and -16.14 ms front; capture-result timestamp errors are 228/226. These negative ages are retained as **FAILED** evidence, never reported as low latency. The capture provider remains stable and isolation passes. [Supported camera run](../../build/candidates/candidate54-road-mean/emulator-camera-supported.json) · [Isolation](../../build/candidates/candidate54-road-mean/emulator-camera-supported-experiment.json) · [Exact additional command](../../build/candidates/candidate54-road-mean/emulator-extra-commands.json)

Final cleanup is **PASSED** on both devices: no app/test processes, no owned cameras, no forward/reverse mappings, unchanged APK identity and at least 30 continuous seconds NORMAL. Fold skin is 27.4°C after 32.28 seconds NORMAL; emulator skin is 30.8°C after 30.79 seconds NORMAL. The emulator is shut down after cleanup. [Fold cleanup](../../build/candidates/candidate54-road-mean/fold-final-cleanup.json) · [Emulator cleanup](../../build/candidates/candidate54-road-mean/emulator-final-cleanup.json) · [Machine-readable status](../../build/candidates/candidate54-road-mean/validation-summary.json)

The initial host run failed four native compilations because macOS selected an incompatible SDK target. The corrected MacOSX26.5 SDK run and final rerun both pass all 224 tests. One candidate53 camera command used unsupported `--report` instead of `--output`; it failed before opening cameras and is preserved separately as a CLI error. Hardware camera rejections and accuracy failures remain separate from those orchestration/environment errors.

The post-edit arity check finds six controls-progress callers and no provably incompatible arities. The broader static quality scan against repository HEAD **FAILS** with 113 gating findings across the accumulated uncommitted port; it also indexes populated submodules and retained reports. This is not a candidate53-to-candidate54 delta and is not presented as clean. Complexity findings include existing large camera/report paths; a broad refactor was not mixed into the measured APK. The MCP check timed out; the CLI completed. [Static-check summary](../../build/candidates/candidate54-road-mean/quality-delta-summary.json)

All experiments stop app/test processes, verify camera release and provider stability, remove experiment-created forwards and require at least 30 continuous seconds of NORMAL thermal status with skin at or below 32°C before starting. No cooling settings were changed. Validation failure does not mean isolation failure; each receipt retains both fields and the actual APK identity.

Isolated accuracy/freshness/timing gates do not pass, so two-minute and 30-minute combined perception workloads are **BLOCKED / NOT RUN**. No sustained 20 Hz perception or qualified 50 ms sensor-to-model-output claim is made.

## Reproduce

Use the provisioned model assets, SDK and staged engine sources, from the repository root. The source/patch identities and exact validation argv are retained beside the APK.

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME="$HOME/Library/Android/sdk"
export PATH="$ANDROID_HOME/platform-tools:$PATH"
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python"
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
export CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
.venv/bin/python android/scripts/build_ncnn.py
android/gradlew -p android --offline --console plain -Pfold3CameraCompat=true \
  -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
.venv/bin/python -m unittest discover -s android/tests

candidate=android/build/candidates/candidate54-road-mean
for s in RFCR81Y1TBJ emulator-5586; do
  adb -s "$s" install -r "$candidate/app-debug.apk"
  adb -s "$s" install -r "$candidate/app-debug-androidTest.apk"
  .venv/bin/python android/reports/full-model-controls-20261101/verify_installed.py \
    --serial "$s" --candidate "$candidate" --output "$candidate/reinstalled-$s.json"
done
```

[Fold validation commands](../../build/candidates/candidate54-road-mean/fold-commands.json) · [Emulator validation commands](../../build/candidates/candidate54-road-mean/emulator-commands.json) · [Campaign runner](../../build/candidates/candidate54-road-mean/run_validation.py). Each command includes the experiment wrapper and default 30-second thermal/cleanup gate. Use fresh output paths to preserve earlier evidence. The model validator blocks timing automatically if original-reference accuracy is incomplete. Camera commands use `--output`; instrumentation and other validators use `--report`.
