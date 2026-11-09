# Candidate71: spatial weight reuse

Diagnostic Android port; TX remains compiled out and physical engagement disabled. Accuracy tolerances remain rtol=1e-3, atol=1e-4; the complete sensor-to-published-model-output maximum remains 50 ms. Cooling optimization is deferred and thermal protections remain active. Earlier candidates and failures are preserved.

## Change

The pinned ncnn Vulkan spatial GEMM shader now loads each packed convolution weight once for four output positions. Four independent accumulators preserve each output's original ky/kx/scalar-channel order, strict FMA correction, bias and rounding boundaries. Input address bases are hoisted outside the inner loop. Tail positions are guarded before reads/writes. Scalar and packed input/output layouts remain supported. Original models and converted graphs are unchanged. The new patch is applied reproducibly after the existing patches; native build metadata and Gradle stale-build rejection include its hash.

The host test compiles the actual changed shader body as C++ and checks bitwise agreement with independent scalar convolution across scalar/packed layouts, 16/24/32 channels, row crossings and partial tiles. It supplements rather than replaces full untapped model validation on the Fold3.

Controls production behavior, pacing and metric definitions are unchanged. The supported 2 ms ultrawide diagnostic compares direct YUV with PRIVATE/GLES at 640x360; this does not qualify illumination, mounting or rear sensor identity. Samsung mode37 is retained as the three-stream transport baseline. No provider-crash exclusions are bypassed.

## Artifacts and accuracy

Main APK `8d2359a5a0b6ccdf61a73e4c7d04bdb2ef2b0e52f09799c4cc99bd032901c762` (1,147,201,452 bytes). Test APK `4bcafe0c4ea192d0b10c409ec32b10fb14cfc2302e04b79669be0d55090bfc7c`. Packaged ncnn `ecfeeae5f9c40594b40753d88945fed4f582b9ebd495374ff305e9833466d65d`. Both installed hashes are verified on Fold3 and the dedicated ARM64 Android15 emulator. Branch codex/fold3-android-port; base HEAD 7631d5248cc9efd10b962107aadacbb908a9f14d.

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

Separate profiled GPU layers (mean ms; not substituted for complete inference):

| Model / layer | Candidate70 | Candidate71 |
|---|---:|---:|
| dmonitoring_model / conv_69 | 5.643 | 5.335 |
| dmonitoring_model / conv_70 | 7.977 | 9.310 |
| dmonitoring_model / conv_72 | 6.714 | 8.018 |
| dmonitoring_model / div_0 | 2.472 | 2.471 |
| driving_supercombo / conv_88 | 17.067 | 24.085 |
| driving_supercombo / conv_114 | 8.976 | 8.974 |
| driving_supercombo / div_1 | 0.526 | 0.643 |

The four-position reuse experiment is rejected as a performance optimization: road conv_88 GPU mean increased from 17.067 to 24.085 ms. Complete inference does not demonstrate a meaningful speed improvement, despite passing accuracy. Candidate70 remains the prior comparison baseline.

Detailed upload, submission, execution and readback evidence remains in fold-profile.json. Diagnostic profiling is excluded from unprofiled distributions. Isolated inference still exceeds 50 ms; sustained perception qualification is not reached.

## Camera timing

| Profile / stream | Acquisition p99 ms | Publication p99 ms | Published / frames | Over 50 ms |
|---|---:|---:|---:|---:|
| fold-ultrawide-manual-2ms-low-latency-yuv / 2 | 54.956 | 55.882 | 357 / 357 | 302 |
| fold-ultrawide-manual-2ms-low-latency-private / 2 | 56.655 | 60.403 | 357 / 358 | 344 |

The recorded ultrawide camera2 characteristics expose regular FPS ranges up to [30,30], a 33,333,333 ns minimum frame duration at 640x360, and no constrained high-speed modes. The PRIVATE trial does not establish a faster supported route.

Samsung three-camera transport passed: 359 composite, 351 rear-preview, 358 front. Rear acquisition/publication p99 92.106/97.519 ms; front delivery 96.420 ms. Freshness FAIL; independent rear identity BLOCKED. Composite pixels and timestamp overlap are insufficient physical identity evidence. Pairing within 20 ms and the complete 50 ms deadline remain unqualified. Sensor capture times are retained; arrival times do not replace invalid timestamps.

Emulator timing: 340 frames, 340 invalid capture timestamps rejected, 0 published. Emulator camera timing failure is retained separately from deterministic injected-input correctness.

## Controls and UI

Matched full-chain 30-second synthetic fixtures; factory ACC state handling, cancellation and stale-input guards retained. No physical Panda transmission.

| Device / fixture | Work p99 ms | Complete-cycle p99 ms | Hz | Scheduled misses / skips | Raw cycle misses | UI fps |
|---|---:|---:|---:|---:|---:|---:|
| fold-steady | 2.291 | 12.294 | 99.996 | 0 / 0 | 1277 | 29.70 |
| fold-faults | 2.104 | 12.455 | 99.999 | 0 / 0 | 1296 | 29.59 |
| emulator-steady | 1.092 | 12.067 | 100.004 | 0 / 0 | 1379 | 29.58 |
| emulator-faults | 1.228 | 11.584 | 99.999 | 0 / 0 | 1274 | 29.61 |

The unchanged complete-cycle timing gate still fails. Short work and scheduled completion do not erase raw misses. No production controls timing fix is claimed. Stage, CPU/off-CPU and raw timing evidence is retained in controls-timing-analysis.json.

## Validation and cleanup

- PASS: 260 focused host tests, patched native build, APK/test APK build; Java unit task up to date from previous successful run.
- Instrumentation fold: passed, 30/30; lifecycle passed.
- Instrumentation emulator: passed, 30/30; lifecycle passed.
- Emulator MetaDrive passed: 2620 packets, 912 inactive after fault, 0 longitudinal active, 0 below-minimum active, maximum lateral error 0.4699 m. Synthetic perception/plant only.
- PASS: all 15 isolation receipts. App/test processes stopped, cameras released, experiment forwarding/reverse removed, hashes/provider PIDs stable; owned emulator shut down. Other connected phone untouched.
- Final fold NORMAL cooldown 32.282 continuous seconds, skin 29.0 C.
- Final emulator NORMAL cooldown 30.766 continuous seconds, skin 30.8 C.
- PASS: 83 immutable artifact and 171 source hashes. Packaged Python assets match candidate70; only reproducible completed staging reclaimed.
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
.venv/bin/python android/build/candidates/candidate71-spatial-reuse/campaign.py
```

The launcher JAVA_HOME does not override the configured Java21 Gradle daemon. The candidate-local resume init file reuses previously verified completed Python assets for this native-only change; ordinary clean builds use the standard command above. Preserve existing receipts before reruns. Device commands are retained in *-command.json and always use experiment.py leases, teardown and 30-second NORMAL cooldowns. Accelerator timing is physical Fold3 evidence; emulator fixtures establish functional recovery only.

[APK](../../build/candidates/candidate71-spatial-reuse/app-debug.apk) · [Manifest](../../build/candidates/candidate71-spatial-reuse/artifacts.json) · [Results](../../build/candidates/candidate71-spatial-reuse/validation-summary.json) · [Model comparisons](../../build/candidates/candidate71-spatial-reuse/fold-models.json) · [GPU profiling](../../build/candidates/candidate71-spatial-reuse/fold-profile.json) · [Controls analysis](../../build/candidates/candidate71-spatial-reuse/controls-timing-analysis.json).
