# Fold 3 accelerator recovery — 7 November 2026

Branch: `codex/fold3-android-port`, based on `7631d5248cc9efd10b962107aadacbb908a9f14d`. Earlier candidates and failed trials remain intact. Vehicle TX is compiled out and physical engagement disabled. Cooling optimization is deferred; thermal safeguards are unchanged.

## Changes

- Batch ncnn Vulkan terminal-output downloads, reuse owned input/output storage and session allocators, and reject unsupported non-input layer assignments. The executing ncnn path uses Vulkan buffers; the adapter's legacy `memory_mode=image` selector does not imply Vulkan image storage.
- Add optional GPU event profiling using the actual queue timestamp width/period, wrap handling and existing fences. Separate host preparation, command recording, final fence wait and host copying. Fence waiting includes GPU execution and transfers; command-recording time can include internal waits. Neither is an independent pure CPU or upload measurement. Profiling is disabled for acceptance/timing, and profiled or tapped graphs cannot create qualification records.
- Tile verified stem and pointwise convolutions while preserving each output's accumulation order. For demonstrably exact short-significand products, use separately rounded multiply/add; retain software FMA correction otherwise. Exhaustive finite-half operand coverage and full original-reference comparisons guard this change. Candidate64 exposed costly false predicate checks in FP32 spatial convolutions; candidate65 restricts that spatial shortcut to the first driver stem and retains the pointwise shortcut for comparison; its earlier gains did not consistently reproduce.
- Hash all shader patches in native-build identity and reject a stale native build from Gradle. Qualification records bind original and converted models, implementation/provider libraries, APK and device; failed timing or CPU fallback cannot produce qualified perception.
- Expand HTP probes to FP16, 16-bit activations/16-bit weights and three 16-bit activation/8-bit weight variants. Calibration seeds 1000–1015 are separate from validation seed 2029 and the original model suite. Reports distinguish original-reference quantization loss from backend execution error against QDQ CPU outputs. Primitive success cannot qualify a complete model.

## Validation rules

Original model hashes, goldens, `rtol=1e-3`, `atol=1e-4`, three seeds (42/7/123), three 32-step recurrent sequences and the pinned 60-frame replay are preserved: 159 driver and 636 road terminal comparisons, 795 overall. Full untapped accuracy precedes timing. All 100 timing samples count, including the first; a missed deadline cannot be hidden by warmup exclusions.

The 50 ms maximum sensor-to-published-output requirement is unchanged. Road timing starts at the earlier current rear capture; driver timing starts at front capture. Isolated inference exceeding 50 ms already fails this requirement. A functional camera or simulator pass does not qualify freshness, independent rear identity, vehicle mounting, or physical actuation.

The controls loop retains one absolute 10 ms release schedule, nonblocking latest-input reads, skipped-slot accounting, cancellation, stale-input checks and watchdogs. No metric definition or threshold was changed. Legacy complete-cycle intervals and misses remain authoritative alongside work and scheduled release-to-completion measurements.

## Reproduce

Use the pinned prerequisites and model/sequence preparation in [Android setup](../../README.md). From the repository root, with the existing prepared original model assets:

```sh
.venv/bin/python android/scripts/build_ncnn.py
env SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest discover -s android/tests

env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home \
  ANDROID_HOME=/Users/sergios/Library/Android/sdk \
  FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python" \
  android/gradlew -p android --offline --console plain \
  -Pfold3CameraCompat=true -Pfold3Qnn=true \
  -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
```

The archived `*-command.json` files contain exact device commands. Run each through `android/scripts/experiment.py`, which takes a per-device lease, stops both application variants, removes experiment forwarding, checks camera/provider release and requires 30 continuous seconds of NORMAL status with skin at or below 32°C. It repeats cleanup after each experiment. No thermal override is used.

For an already installed matching APK, the archived campaign commands are:

```sh
.venv/bin/python android/build/candidates/candidate65-scoped-fma/run_validation.py \
  fold models profile npu instrumentation controls-steady controls-faults lifecycle
.venv/bin/python android/build/candidates/candidate65-scoped-fma/run_validation.py \
  emulator instrumentation controls-faults lifecycle controls-steady metadrive
.venv/bin/python android/build/candidates/candidate65-scoped-fma/run_cameras.py fold
.venv/bin/python android/build/candidates/candidate65-scoped-fma/run_cameras.py emulator
```

Only measure accelerator accuracy and speed on the physical Fold 3. Emulator fixtures exercise lifecycle, IPC, injected input and recovery; real emulator camera timing is a separate test.

## Results

Final artifact: **candidate65**, APK SHA-256 `8170e642d4c002f84fe2c1cb95ae79505a11be7e4016c707aa0cc3e6cc84b7ec`. The installed hash matches on RFCR81Y1TBJ (SM-F926B, Android 15) and the ARM64 Android 15 emulator. Test APK SHA-256: `bcc20fc1b227cb03bde6cf0c6fbc153dce82eb718ffb50cf6fd6fad45b6e999c`.

[Installable APK](../../build/candidates/candidate65-scoped-fma/app-debug.apk) · [artifact/native/model/source hashes](../../build/candidates/candidate65-scoped-fma/artifacts.json) · [machine-readable summary](../../build/candidates/candidate65-scoped-fma/qualification-summary.json) · [comparison and exact commands](../../build/candidates/candidate65-scoped-fma/)

| Gate | Result | Evidence |
|---|---|---|
| Host tests | PASSED | 250 tests; original precision, preprocessing, IPC, scheduling, supervision, Hyundai and receive-only checks |
| Build and Java unit tests | PASSED | NDK 28.2, API 35, pinned engines, Gradle debug and instrumentation APKs |
| Full untapped Vulkan accuracy | PASSED | 795/795; original seeds, recurrent sequences and replay; unchanged tolerances; no CPU fallback or unexpected non-finite outputs |
| Instrumentation | PASSED | 30 tests on each device; lifecycle, supervision, numerical/thermal faults, IPC and USB rejection |
| Functional fixtures | PASSED | Controls chain, stale inputs, factory-ACC cancellation; emulator display recovery and MetaDrive |
| HTP primitive qualification | FAILED/BLOCKED | 8 passed, 6 accuracy failures, 6 unsupported/initialization blocks; no full model promoted |
| Inference deadline | FAILED | Both models exceed 50 ms on 100/100 samples; observed maximum driver 290.332 ms, road 458.114 ms |
| Complete controls cycle | FAILED | See raw counters below; scheduled release completion is a separate metric |
| Camera freshness | FAILED | Actual sensor-to-consumer measurements below, before inference |
| Independent rear identity | BLOCKED | Composite delivery is insufficient proof; manual optical trials remain deferred |
| Sustained perception workload | BLOCKED | No 30 s / 2 min / 30 min combined-perception progression past failed prerequisite gates |
| Mounted calibration / physical Panda / vehicle | BLOCKED | Phone unmounted, Panda disconnected; no TX authorization or physical actuation |

### Matched isolated model workloads

Milliseconds; p50 / p99. All candidates below pass 795 comparisons; each model misses 50 ms on all 100 timing samples. These are observed runs, not guaranteed patch-attributable gains; clocks were not locked and results vary across runs.

| Candidate | Driver p50 / p99 | Road p50 / p99 |
|---|---:|---:|
| [candidate60](../../build/candidates/candidate60-road-rounding/artifacts.json) | 173.366 / 181.884 | 319.175 / 355.129 |
| [candidate62](../../build/candidates/candidate62-gpu-batchfix/artifacts.json) | 174.407 / 182.698 | 319.046 / 460.717 |
| [candidate63](../../build/candidates/candidate63-vulkan-tiles/artifacts.json) | 188.000 / 230.991 | 303.143 / 426.306 |
| [candidate64](../../build/candidates/candidate64-exact-product/artifacts.json) | 190.510 / 198.727 | 308.144 / 382.896 |
| [candidate65](../../build/candidates/candidate65-scoped-fma/artifacts.json) | 187.602 / 198.759 | 303.183 / 430.381 |

Candidate60 remains the accurate measured baseline. Candidate65 road median improves 4.97% versus candidate62, but driver median regresses 7.57% and driver p99 regresses 8.79%. This is **not an overall accepted optimization**; no live backend promotion or deadline qualification was granted. Candidate61’s premature packed-output rejection is retained as a failed trial and was fixed in candidate62. Earlier performance failures are preserved.

Diagnostic component gains did not consistently reproduce in the complete model. Candidate64 measured driver conv69 at 7.12 ms versus 15.17 ms and road conv96 at 4.20 ms versus 10.23 ms in candidate63. Candidate65 retains driver conv69 at 7.64 ms, but road conv96 measures 11.13 ms. Its road stem conv88 remains 38.05 ms and is the largest measured GPU operator. These diagnostic profiles are excluded from acceptance timing; they do not establish a stable end-to-end speedup.

HTP FP16 and 16-bit-weight Conv/Gemm remain unsupported by the tested assignment/initialization paths. Quantized Conv/Gemm fail original tolerances; Conv also differs from the QDQ CPU result, while tested Gemm backend results match QDQ CPU and still fail the original unquantized graph. More nominal accelerator throughput cannot clear these numerical failures. No silent CPU fallback is allowed.

### Controls

Same full-chain fixture, timing schema 2. Pacing, input handling, computation, IPC, publishing and reporting remain measured; counters were not reset or redefined to clear a failure.

| Device/workload | Work p99 ms | Complete cycle p99 ms | Hz | Scheduled misses / skipped slots | Raw cycle misses |
|---|---:|---:|---:|---:|---:|
| fold-controls-steady | 2.282 | 12.266 | 99.994 | 0 / 0 | 1155 |
| fold-controls-faults | 2.157 | 12.095 | 99.997 | 0 / 0 | 1250 |
| emulator-controls-steady | 1.146 | 11.920 | 99.996 | 0 / 0 | 1349 |
| emulator-controls-faults | 1.636 | 11.569 | 100.008 | 0 / 0 | 1321 |

Work and absolute scheduled completion pass in these fixtures, but the legacy complete-cycle gate remains failed. This turn did not change controls calculations or clear that failure through a new metric. These synthetic fixtures are not a concurrent, qualified camera/inference workload.

### Camera measurements

- Fold 3 public ultrawide 640×360 direct YUV: 356 frames, delivery p99 83.615 ms, acquisition p99 82.867 ms. Transport success does not pass 50 ms freshness.
- Samsung mode 37 three-stream baseline: 352 preview, 357 composite and 359 front frames. Preview acquisition p99 111.699 ms, preview sink delivery p99 116.886 ms, front delivery p99 97.035 ms. Transport status passed; rear physical identity remains unproven and VisionIPC is unqualified.
- Emulator advertised camera 1 / 640×480: 343 frames, 343 invalid capture-result timestamps and 0 published frames. Future capture timestamps are rejected, not replaced by arrival time. Deterministic injected fixture success does not clear this provider timing fault.

All camera timestamps, crops, origins, counts and readback/publication stages are in the raw camera reports. Rear pairing still requires 20 ms and the road deadline begins at the earlier current rear capture; no composite or simulator result can establish independent sensor identity.

### Cleanup and remaining work

All 15 final-candidate experiment isolation receipts pass. Each required at least 30 continuous seconds of NORMAL thermal status and skin ≤32°C. Final cleanup on both devices verifies the installed APK hash, stopped app/test processes, no camera ownership, unchanged camera-provider processes, no experiment forward/reverse mappings and a fresh cool baseline. The owned emulator is shut down; the phone is left stopped. Only regenerable Gradle asset copies were removed on the Mac, preserving source, original models, APKs and evidence.

All 163 archived source hashes still match the worktree. The whole-working-tree quality scan is retained in `working-tree-quality-delta.json`; it reports complexity/size findings across the existing uncommitted port and historical reports, so no clean quality-gate claim is made. Earlier failed host-SDK, emulator-storage, packed-output and camera-configuration attempts are retained rather than overwritten.

Remaining implementation blockers are accurate accelerated inference within 50 ms, a stable independent three-camera route within the capture budget, and the existing complete-cycle timing gate. Physical rear proof, mounted optical/vehicle calibration, Panda firmware/passive fingerprint/harness checks and real vehicle behavior remain separate unavailable-hardware gates. TX stays compiled out regardless of bench results.
