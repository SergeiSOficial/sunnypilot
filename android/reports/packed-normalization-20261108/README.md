# Candidate70: packed Vulkan normalization and camera comparison

Candidate70 fixes the missing packed shader coverage and preserves all 795 original-reference comparisons. Driver inference improves modestly in a matched short comparison. Road speed is not consistently better than candidate68. Inference, three-camera freshness and the unchanged controls complete-cycle gate still fail. This is a diagnostic APK, not a qualified driving port. TX remains compiled out and physical engagement disabled. Cooling optimization is deferred; all thermal safeguards remain active.

## Implementation and artifacts

The exact byte-normalization helper now covers both scalar and pack4 BinaryOp Vulkan shaders. Inspection of pinned ncnn VkCompute::record_upload shows input dimensions divisible by four are uploaded as pack4 independently of the CPU packing-layout option. Candidate69 covered only the scalar shader and had no meaningful speedup. Its complete measurements and failures remain in the [candidate69 report](../exact-normalization-20261108/README.md).

Only scalar division by constant 255 uses the shortcut, and only for finite integral values 0..255. Exact integer significand construction preserves FP32 rounding, including negative zero; other inputs retain strict reference division. Non-scalar division, other divisors and reduced-precision paths are unchanged. The host check compiles the actual helper, exhausts all 256 values and checks fallback edges; it also verifies the scalar and packed helper bodies agree and all four packed lanes are covered. Original models, graphs, casts, accumulation order and tolerances are unchanged.

The pinned build recipe and Gradle stale-build checks bind normalization and dispatch patches to the generated library. Qualification records continue to bind actual native-library hashes. No CPU fallback can qualify perception.

Main APK SHA256 `9ef37ce35eabdafdba80c0039b8bbfc4425068152c923c461af0f0f5bbe61db8` (1,147,201,452 bytes).

Test APK `4bcafe0c4ea192d0b10c409ec32b10fb14cfc2302e04b79669be0d55090bfc7c`.

Packaged libncnn `5f0d7e8f1d856ca66d2a26d26a65991ec4de7d0bc6ef2f2571d1ece68e8ca204`.

Both main/test installed hashes match on the SM-F926B and dedicated ARM64 Android15 emulator. Branch: codex/fold3-android-port, base HEAD 7631d5248cc9efd10b962107aadacbb908a9f14d. The existing workspace and frozen earlier candidates are preserved.

## Model accuracy and isolated timing

PASS: all 795 comparisons at rtol=1e-3, atol=1e-4, including three seeds, recurrent sequences and the pinned replay, with no unexpected non-finite outputs or CPU fallback. Complete untapped models pass before timing. Timing is 100 unprofiled samples per model; the separate GPU profile is diagnostic and excluded from these distributions.

| Candidate / model | p50 ms | p99 ms | Maximum ms | Over 50 ms |
|---|---:|---:|---:|---:|
| 68 / driver | 144.506 | 154.106 | 181.566 | 100/100 |
| 68 / road | 328.765 | 362.394 | 362.635 | 100/100 |
| 69 / driver | 144.227 | 153.467 | 176.957 | 100/100 |
| 69 / road | 343.210 | 362.294 | 362.994 | 100/100 |
| 70 / driver | 138.717 | 149.742 | 168.531 | 100/100 |
| 70 / road | 335.224 | 356.939 | 362.867 | 100/100 |

FAIL: every isolated inference exceeds the complete 50 ms sensor-to-published-output deadline before camera delay. Short-run gains cannot qualify sustained 20 Hz or the full pipeline. Road median remains worse than candidate68; no overall road performance promotion is claimed.

Separate GPU measurements:

- dmonitoring_model: top measured layers `[{"node":"conv_70","operator":"Convolution","mean_gpu_ms":7.9766248052581785},{"node":"conv_72","operator":"Convolution","mean_gpu_ms":6.7144269194068915},{"node":"conv_99","operator":"Convolution","mean_gpu_ms":5.7222238186306}]`.
- driving_supercombo: top measured layers `[{"node":"conv_88","operator":"Convolution","mean_gpu_ms":17.066536041669846},{"node":"conv_114","operator":"Convolution","mean_gpu_ms":8.975697697533416},{"node":"conv_109","operator":"Convolution","mean_gpu_ms":5.413395701170349}]`.

Driver division profile entries: `[{"node":"div_0","operator":"BinaryOp","mean_gpu_ms":2.4715624396591185}]`. Detailed submission, transfer and readback records remain in fold-profile.json. These profiled components cannot be summed across independent percentile distributions to clear the timing gate.

## Controls and UI

No production controls policy, calculations, metric definitions or raw counters change in this candidate. Both devices rerun the matched synthetic full-chain fixture, including cancellation, stale inputs and factory ACC state handling. Fixture SHA256 b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99. Synthetic perception and vehicle dynamics do not qualify real driving.

| Device / fixture | Work p99 ms | Complete-cycle p99 ms | Mean Hz | Scheduled misses / skipped | Raw cycle misses | UI fps |
|---|---:|---:|---:|---:|---:|---:|
| fold-steady | 2.177 | 12.301 | 99.994 | 0 / 0 | 1312 | 29.73 |
| fold-faults | 2.196 | 12.437 | 99.997 | 0 / 0 | 1237 | 29.59 |
| emulator-steady | 1.218 | 11.187 | 100.006 | 0 / 0 | 1283 | 29.57 |
| emulator-faults | 1.304 | 12.180 | 99.997 | 0 / 0 | 1327 | 29.57 |

Functional controls and work-time checks pass. The unchanged complete-cycle gate still fails; no controls timing fix is claimed. Absolute slot pacing, nonblocking latest inputs, skipped-slot accounting, cancellation, stale-input guards, heartbeat and watchdog/reporting-failure checks remain intact. Raw stage, CPU/off-CPU, wakeup and complete-cycle data are retained in controls-timing-analysis.json and the JSONL captures.

## Camera-only results

The new supported manual-2ms-low-latency profile is diagnostic, not the live camera default. Camera 2 uses direct YUV 640x360 with advertised 30 fps frame duration and capability/session/exposure checks. This does not qualify illumination, independent rear identity or mounting geometry. Both trials have clean baselines and unchanged capture-time accounting.

| Candidate / profile | Acquisition p99 ms | Publication p99 ms | Publications | Over 50 ms |
|---|---:|---:|---:|---:|
| 69 / low-latency-yuv | 72.355 | 73.903 | 357 | 357 |
| 69 / manual-yuv | 60.049 | 61.894 | 357 | 357 |
| 69 / manual-low-latency-yuv | 60.170 | 61.675 | 357 | 357 |
| 69 / manual-2ms-low-latency-yuv | 53.784 | 55.873 | 358 | 298 |
| 70 / manual-2ms-low-latency-yuv | 54.544 | 55.674 | 357 | 305 |

Three-camera Samsung mode37 baseline: 359 composite, 351 rear-preview and 359 front frames. Rear texture acquisition p99 99.782 ms, publication 105.184 ms; front delivery 98.568 ms. Transport passed; freshness FAIL. Independent rear identity remains BLOCKED: composite transport, pixel differences and overlapping timestamps do not establish independently assigned physical outputs or repeatable optical evidence. Rear pairing within 20 ms and the complete 50 ms deadline remain unqualified. No timestamps or input rejections are hidden.

Emulator camera timing FAIL: 342 frames, 342 rejected capture timestamps, 342 capture-result timestamp errors and 0 publications. Invalid/negative ages are clock-fault diagnostics, not camera speed. Arrival times never substitute for invalid capture times.

## Validation and cleanup

- PASS: 258 focused host tests; patched native build; APK/test APK build; Java unit check up to date from the successful candidate69 run; instrumentation 30/30 on each device. Lifecycle recovery and emulator display changes pass. Physical hinge movement was not exercised.
- RETAINED FAILURE: the initial Fold3 suite failed 1/30 because its manual rear-exposure assertion required lowercase text while the correct rejection starts with a capital letter. The test now compares the message case-insensitively, retaining the blocked-status, specific error and zero-open-camera assertions. Both devices rerun all 30 tests with the revised test APK; the production APK stays byte-identical. The original test APK, manifest and failed receipts remain archived. Candidate69's report also retains this failure.
- PASS: unchanged Hyundai behavior/safety source checked in this work session: 13 run / 2 skipped and clean final safety run 1899 / 198 skipped, respectively. Those host receipts belong to candidate69's unchanged relevant source; they are not physical Panda evidence. Earlier failed/interrupted logs remain retained.
- PASS: emulator MetaDrive stale/cancel recovery: 2618 packets, 916 inactive-after-fault, 0 longitudinal-active, 0 below-minimum active, max lateral error 0.5524 m. Synthetic perception and plant only.
- PASS: all 16 isolation receipts; processes stopped, cameras released, forwards/reverse mappings removed, installed hashes/provider PIDs stable across experiments. Both final cleanup checks pass and the owned emulator shuts down. The other connected phone is untouched.
- Final fold cool NORMAL baseline after the instrumentation rerun: 32.340 continuous seconds, skin 29.1 C. Installed main/test hashes match again, processes/cameras/forwarding are clear; the second owned emulator session also shuts down cleanly.
- Final emulator cool NORMAL baseline after the instrumentation rerun: 30.758 continuous seconds, skin 30.8 C. Installed main/test hashes match again, processes/cameras/forwarding are clear; the second owned emulator session also shuts down cleanly.
- PASS: 84 immutable artifact and 169 source hashes.
- PASS: all 20 packaged Python assets are byte-identical to candidate69, APK ZIP integrity passes. Gradle used the verified completed Python asset snapshot while rebuilding the patched native library; the candidate-local resume init script and asset receipts are retained. Only reproducible staging and obsolete unpatched native intermediates were reclaimed. Frozen evidence/models are preserved.
- BLOCKED: repository-wide Ripwire quality delta did not respond in the earlier bounded attempt in this work session; no clean quality-delta result is claimed.
- NOT RUN: qualified combined 30-second, two-minute and 30-minute perception progression because isolated inference/freshness and complete-cycle gates fail. The 30-second synthetic controls fixtures are separate functional checks.
- BLOCKED: independent rear-lens proof, mounted calibration, real Panda firmware/fingerprint/harness and physical vehicle behavior. Phone unmounted; Panda disconnected. TX and engagement remain disabled regardless of bench results. No cooling optimization or thermal safeguard relaxation.

## Reproduce and inspect

From repository root, with the pinned local toolchains/assets and sufficient free space for clean staging:

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
```

The Gradle launcher setting above does not override the repository's configured Java21 daemon. Current-artifact device reruns require exclusive ownership of the dedicated AVD and a connected, unlocked Fold3:

```sh
.venv/bin/python android/build/candidates/candidate70-packed-normalization/campaign.py
```

Preserve/rename existing receipts before a rerun. Every experiment command is retained in its own command JSON and must use experiment.py so its lease, cleanup, timestamp/provider checks and continuous cooldown remain enforced. Do not run the later combined progression while prerequisite gates fail.

[Installable APK](../../build/candidates/candidate70-packed-normalization/app-debug.apk) · [Artifact manifest](../../build/candidates/candidate70-packed-normalization/artifacts.json) · [Compact results](../../build/candidates/candidate70-packed-normalization/validation-summary.json) · [Full-model results](../../build/candidates/candidate70-packed-normalization/fold-models.json) · [GPU profile](../../build/candidates/candidate70-packed-normalization/fold-profile.json) · [Controls analysis](../../build/candidates/candidate70-packed-normalization/controls-timing-analysis.json).
