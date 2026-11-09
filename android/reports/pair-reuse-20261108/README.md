# Candidate72: paired spatial weight reuse

Diagnostic Android port; TX remains compiled out and physical engagement disabled. Accuracy tolerances remain rtol=1e-3, atol=1e-4; the complete sensor-to-published-model-output maximum remains 50 ms. Cooling optimization is deferred and thermal protections remain active. Earlier candidates and failures are preserved.

## Change

The pinned ncnn Vulkan spatial GEMM shader now loads each packed convolution weight once for two output positions. Two bounded pairs retain four outputs per invocation while reducing the number of simultaneously live corrected accumulators. Each output preserves its original ky/kx/scalar-channel order, strict FMA correction, bias and rounding boundaries. Input address bases are hoisted outside the inner loop. Tail positions are guarded before reads/writes. Scalar and packed input/output layouts remain supported. Original models and converted graphs are unchanged. The new patch is applied reproducibly after the existing patches; native build metadata and Gradle stale-build rejection include its hash.

The host test compiles the actual changed shader body as C++ and checks bitwise agreement with independent scalar convolution across scalar/packed layouts, 16/24/32 channels, row crossings and partial tiles. It supplements rather than replaces full untapped model validation on the Fold3.

Controls production behavior, pacing and metric definitions are unchanged. The supported 2 ms ultrawide diagnostic compares direct YUV with PRIVATE/GLES at 640x360; this does not qualify illumination, mounting or rear sensor identity. Samsung mode37 is retained as the three-stream transport baseline. No provider-crash exclusions are bypassed.

Candidate71 four-position reuse is retained as a failed optimization: road conv_88 GPU mean regressed from 17.067 to 24.085 ms. This candidate tests the smaller live accumulator set rather than promoting that regression.

## Artifacts and accuracy

Main APK `dd05f2f5236b1bed3a414ea17a6e1fabdec5dc7af04688fdb12253cac8f169a7` (1,147,201,452 bytes). Test APK `4bcafe0c4ea192d0b10c409ec32b10fb14cfc2302e04b79669be0d55090bfc7c`. Packaged ncnn `17d43b0665fd6206f2d7c3dc117305fad88781fcf262379929aa912ce2a2866b`. Both installed hashes are verified on Fold3 and the dedicated ARM64 Android15 emulator. Branch codex/fold3-android-port; base HEAD 7631d5248cc9efd10b962107aadacbb908a9f14d.

Accuracy: PASS, 795/795 original-reference comparisons. No CPU fallback or unexpected non-finite output is accepted. Seeds, recurrent sequences and pinned 60-frame replay are preserved.

## Isolated inference timing

Unprofiled complete inference, 100 samples/model after accuracy. These exclude camera latency and cannot clear the sensor-to-publication gate.

| Candidate / model | p50 ms | p99 ms | Max ms | Over 50 ms |
|---|---:|---:|---:|---:|
| 68 / dmonitoring_model | 144.506 | 154.106 | 181.566 | 100/100 |
| 68 / driving_supercombo | 328.765 | 362.394 | 362.635 | 100/100 |
| 70 / dmonitoring_model | 138.717 | 149.742 | 168.531 | 100/100 |
| 70 / driving_supercombo | 335.224 | 356.939 | 362.867 | 100/100 |
| 71 / dmonitoring_model | 140.119 | 150.972 | 176.019 | 100/100 |
| 71 / driving_supercombo | 332.290 | 364.663 | 366.496 | 100/100 |
| 72 / dmonitoring_model | 137.223 | 148.841 | 167.380 | 100/100 |
| 72 / driving_supercombo | 311.609 | 363.188 | 363.726 | 100/100 |

Separate profiled GPU layers (mean ms; not substituted for complete inference):

| Model / layer | Candidate70 | Candidate72 |
|---|---:|---:|
| dmonitoring_model / conv_69 | 5.643 | 5.352 |
| dmonitoring_model / conv_70 | 7.977 | 8.272 |
| dmonitoring_model / conv_72 | 6.714 | 7.070 |
| dmonitoring_model / div_0 | 2.472 | 2.472 |
| driving_supercombo / conv_88 | 17.067 | 15.040 |
| driving_supercombo / conv_114 | 8.976 | 8.976 |
| driving_supercombo / div_1 | 0.526 | 0.456 |

Paired reuse reduces the targeted road conv_88 GPU mean from 17.067 to 15.040 ms in separate profiling. Unprofiled road median falls from 335.224 to 311.609 ms (7.0 percent) in this workload, while p99 increases from 356.939 to 363.188 ms. Driver median/p99 are 137.223/148.841 ms. This is limited median/kernel progress, not a passing complete-inference or sustained-performance result.

Detailed upload, submission, execution and readback evidence remains in fold-profile.json. Diagnostic profiling is excluded from unprofiled distributions. Isolated inference still exceeds 50 ms; sustained perception qualification is not reached.

## Camera timing

| Profile / stream | Acquisition p99 ms | Publication p99 ms | Published / frames | Over 50 ms |
|---|---:|---:|---:|---:|
| fold-ultrawide-manual-2ms-low-latency-yuv / 2 | 54.334 | 56.001 | 357 / 357 | 304 |
| fold-ultrawide-manual-2ms-low-latency-private / 2 | 56.708 | 60.890 | 357 / 358 | 340 |

The recorded ultrawide camera2 characteristics expose regular FPS ranges up to [30,30], a 33,333,333 ns minimum frame duration at 640x360, and no constrained high-speed modes. The PRIVATE trial does not establish a faster supported route.

Samsung three-camera transport passed: 360 composite, 352 rear-preview, 359 front. Rear acquisition/publication p99 99.460/104.056 ms; front delivery 97.442 ms. Freshness FAIL; independent rear identity BLOCKED. Composite pixels and timestamp overlap are insufficient physical identity evidence. Pairing within 20 ms and the complete 50 ms deadline remain unqualified. Sensor capture times are retained; arrival times do not replace invalid timestamps.

Emulator timing: 342 frames, 342 invalid capture timestamps rejected, 0 published. Emulator camera timing failure is retained separately from deterministic injected-input correctness.

## Controls and UI

Matched full-chain 30-second synthetic fixtures; factory ACC state handling, cancellation and stale-input guards retained. No physical Panda transmission.

| Device / fixture | Work p99 ms | Complete-cycle p99 ms | Hz | Scheduled misses / skips | Raw cycle misses | UI fps |
|---|---:|---:|---:|---:|---:|---:|
| fold-steady | 2.141 | 12.142 | 99.999 | 0 / 0 | 1231 | 29.72 |
| fold-faults | 2.222 | 12.136 | 100.000 | 0 / 0 | 1271 | 29.52 |
| emulator-steady | 1.299 | 12.151 | 100.008 | 0 / 0 | 1315 | 29.60 |
| emulator-faults | 1.814 | 11.947 | 99.998 | 0 / 0 | 1270 | 29.58 |

The unchanged complete-cycle timing gate still fails. Short work and scheduled completion do not erase raw misses. No production controls timing fix is claimed. Stage, CPU/off-CPU and raw timing evidence is retained in controls-timing-analysis.json.

## Validation and cleanup

- PASS: 260 focused host tests, patched native build, APK/test APK build; Java unit task up to date from previous successful run.
- Instrumentation fold: passed, 30/30; lifecycle passed.
- Instrumentation emulator: passed, 30/30; lifecycle passed.
- Emulator MetaDrive passed: 2626 packets, 915 inactive after fault, 0 longitudinal active, 0 below-minimum active, maximum lateral error 0.5034 m. Synthetic perception/plant only.
- PASS: all 15 isolation receipts. App/test processes stopped, cameras released, experiment forwarding/reverse removed, hashes/provider PIDs stable; owned emulator shut down. Other connected phone untouched.
- Final fold NORMAL cooldown 32.346 continuous seconds, skin 29.6 C.
- Final emulator NORMAL cooldown 30.748 continuous seconds, skin 30.8 C.
- PASS: 83 immutable artifact and 171 source hashes. All 20 packaged Python assets match candidate71. Completed verified staging is retained, and build daemons are stopped before experiments.
- RETAINED: the first candidate72 packaging attempt omitted Chaquopy build.json. The pre-install audit rejected it; it was never installed. Regenerating the standard metadata task restored all 20 matching Python assets. The rejected APK hash and reason remain in prepackage-rejection.json.
- BLOCKED: Ripwire quality delta returned no result; no clean repository-wide quality result claimed. Source mapping completed; git diff --check passes.
- NOT RUN: qualified combined 30-second, two-minute and 30-minute progression because isolated inference/freshness and complete-cycle gates fail. Synthetic controls fixtures are functional checks, not sustained perception qualification.
- BLOCKED: rear optical identity proof, mounted calibration and real Panda firmware/fingerprint/harness/vehicle behavior. Phone unmounted, Panda disconnected. TX and engagement remain disabled. Thermal pauses remain active; cooling optimization deferred.

## Reproduce

From repository root, with pinned local assets/toolchains and sufficient disk space for clean Python staging:

```sh
.venv/bin/python android/scripts/build_ncnn.py
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest discover -s android/tests
JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
ANDROID_HOME=/Users/sergios/Library/Android/sdk \
FOLD3_BUILD_PYTHON=$PWD/.venv/bin/python \
  android/gradlew -p android --offline --console plain \
  -Pfold3CameraCompat=true -Pfold3Qnn=true \
  -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
.venv/bin/python android/build/candidates/candidate72-pair-reuse/campaign.py
```

The launcher JAVA_HOME does not override the configured Java21 Gradle daemon. The candidate-local resume init file reuses previously verified completed Python assets for this native-only change; ordinary clean builds use the standard command above. Preserve existing receipts before reruns. Device commands are retained in *-command.json and always use experiment.py leases, teardown and 30-second NORMAL cooldowns. Accelerator timing is physical Fold3 evidence; emulator fixtures establish functional recovery only.

[APK](../../build/candidates/candidate72-pair-reuse/app-debug.apk) · [Manifest](../../build/candidates/candidate72-pair-reuse/artifacts.json) · [Results](../../build/candidates/candidate72-pair-reuse/validation-summary.json) · [Model comparisons](../../build/candidates/candidate72-pair-reuse/fold-models.json) · [GPU profiling](../../build/candidates/candidate72-pair-reuse/fold-profile.json) · [Controls analysis](../../build/candidates/candidate72-pair-reuse/controls-timing-analysis.json).
