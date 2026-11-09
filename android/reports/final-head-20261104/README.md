# Original-reference recovery, 2026-11-04

**Diagnostic work; backend not qualified. TX compiled out; engagement disabled.**
Original models, goldens and `rtol=1e-3`, `atol=1e-4` are preserved. Physical Panda, mounted calibration and independent rear sensor identity remain blocked. Cooling optimization is deferred; thermal guards and experiment cleanup remain enforced.

## Changes and accuracy evidence

- Ordered compensated FP32 packed InnerProduct for ReLU heads, with bias after reduction. Same-input head trace improved exact agreement from 441/624 to 593/624 values; the seed42 untapped result still failed one element. Smaller heads fcrelu16/17 became fully exact locally. `ordered-head-*.json` bind actual provider/probe hashes separately from installed candidate45zq; these are not candidate45zr APK results.
- Added `compare_ncnn_frozen.py`: verifies frozen original host graph reproduces the golden exactly, disables additional fusion and rejects any diagnostic tap changing the final reference output. Missing verified tensor/layout mappings are explicit. The earlier portable tapped reference could hide FP16 boundary differences by changing fusion; it remains useful for mapping, not original-model qualification. Five focused regression tests pass.
- Corrected all three original NHWC 3x3 stride2 spatial convolutions on the actually selected GEMM shader path, preserving bias-first ordered accumulation. Previous rejected experiment corrected only two and its regression is retained in `../remaining-seven-20261104/`.
- Three-spatial **untapped** driver results: seed42 zero mismatches; seed7 zero; seed123 four (indices22,260,282,323). See `three-spatial-{untapped,seed7,seed123}.json`. **Not full accuracy acceptance.** These native probes ran with candidate45zq installed, with separate provider identity recorded.
- A further diagnostic selects the existing ordered FP32 packed head kernel instead of chunked sum8 reduction for larger FP32 ReLU heads. Seed123 still has the same four full-model mismatches; candidate45zs remains immutable. All 45 captured late operators are bit-exact with the same inputs (`large-head-seed123-same-input.json`); frozen-original tracing showed the error propagates from the backbone.

- Located the first captured discrepancy at `tanh_39`: seed123 index215601 differed by one FP32 ULP. The compensated FMA discarded a nonzero correction tail at a rounding midpoint. The scalar tanh FMA now preserves that tail and corrects midpoint selection. `first-tanh-scalar.json` and `tanh-scalar-arithmetic-detailed.txt` preserve the diagnosis.
- **PASS** midpoint native untapped driver seed123: all553 outputs bit-exact (`midpoint-seed123.json`). This probe used a separately hashed provider with candidate45zt installed; it is not a candidate45zu APK result. Full seed/recurrent/replay gate pending.

## Artifacts

- candidate45zr: head correction plus controls priority change, before three-spatial correction. APK SHA256 `56da31ac3c415e77b3541ef662e4843020fa2e812111d09d7fbfcb1a0eb639a7`. Archived, not installed/tested.
- candidate45zs: adds three-spatial correction. APK SHA256 `29c4534a42c5feea779a4d9ec8cd6ffc4577ede8cc74b541ef750cc30bf3cc44`. Full artifacts and provider hashes in `../../build/candidates/candidate45zs/artifacts.json`.
- candidate45zt: larger ordered head selection and clear conversion rejection; installed original-reference run retained, then superseded.
- candidate45zu: adds scalar tanh midpoint correction. APK SHA256 `acfb13ca4ac9e1baf280e364d2b796f97529c3c085a0c0a9120b89bf46967dd5`. Installed phone hash verified.
- Earlier candidates35/40/44 and45zq preserved.

## Controls

Public Android AUDIO thread priority (-16) replaces URGENT_DISPLAY (-8) for the controls thread only; SCHED_OTHER remains in use. Matched 30-second synthetic full-chain comparison (`controls-comparison-zu.json`): work p99 2.693 -> 2.170ms; complete cycle p99 13.561 -> 12.298ms; scheduler delay p99 3.058 -> 1.264ms; release-to-completion p99 5.821 -> 3.974ms; scheduled misses 2 -> 0, skipped slots0, work deadline misses0. Mean100.002Hz. Raw-cycle misses1298 remain: not a complete timing clearance. This is one short bench comparison, not sustained qualification. Absolute single 10ms pacing, stale input/cancellation, skipped slots, watchdogs, raw cycle intervals and all deadline counters remain intact. No metric redefinition clears raw-cycle failure. Baseline45zq: work p99 2.842ms, complete cycle p99 14.597ms.

## Validation so far

- **PASS** 202 focused host tests (`host-midpoint-tests.log`), including the extracted scalar shader against native fmaf for the captured regression, both signs and 200000 additional cases. Earlier compiler-test harness failures are retained in `fma-midpoint-host*.log`. New frozen-trace files pass ruff (`ruff-final.log`). Broader lint retains eight existing issues (UTC `time.time` policy and one dictionary comprehension), recorded in `ruff-complete.log`; not a clean repository-wide lint claim.
- **PASS** native build, Gradle unit tests and APK/test APK builds (`native-build.log`, `three-spatial-native.log`, `gradle-three-spatial.log`).
- **FAIL (retained earlier candidates)** seed123 driver accuracy. The midpoint native probe now passes this seed, but 795-comparison qualification is not achieved; no sustained accelerator performance claim.
- **PASS** candidate45zu Fold3 instrumentation28/28 and 30-second synthetic steady/fault full-chain fixtures, all isolation receipts passed.
- **PASS** candidate45zu installed driver initial seeds42/7/123 all bit-exact. **FAIL** nine frames in recurrent/replay sets: seed42 frames8/21; seed7 frame2; seed123 frames3/4/5; replay frames21/28/59. `phone-candidate45zu-models.json` retains all outputs. **BLOCKED** road missing verified ncnn conversion; existing conversion audit lists unsupported ops/rank. No accelerator timing advanced.
- Emulator reruns pending; previous failures remain in the preceding report.

## Reproduce

From repository root:

```sh
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python"
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
export CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
.venv/bin/python -m unittest discover -s android/tests
.venv/bin/python android/scripts/build_ncnn.py
android/gradlew -p android --no-daemon --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
.venv/bin/python android/scripts/probe_ncnn.py --serial RFCR81Y1TBJ --adb "$HOME/Library/Android/sdk/platform-tools/adb" --seed 123 --trace-count 0 --report /tmp/unique-fold3-probe.json
.venv/bin/python android/scripts/compare_ncnn_trace.py /path/to/trace.json
.venv/bin/python android/scripts/compare_ncnn_frozen.py /path/to/trace.json
```

Each probe stops applications, verifies camera release and forwarding cleanup, and waits for 30 continuous seconds NORMAL thermal status and skin <=32C before/after. Timing fixtures must run without host builds or numerical checks. `run_accuracy.py` and `run_devices.py` record APK identity and sequential experiment isolation; avoid overwriting existing receipts.

## Subsequent recovery, 2026-11-05

- **PASS** all 208 focused host tests (`host-dispatch-full-tests.log`). Shader arithmetic regression checks exercise captured FMA and cube cases plus 200000 FMA / 100000 compensated cube inputs. These host tests do not qualify GPU execution.
- Common FMA midpoint-tail repair cleared all three 32-frame driver sequences. Installed candidate45zv passed 158/159 driver comparisons; replay21 retained one mismatched element. Its APK SHA256 is `d5e1992ef2b2b395c1711813c0ae9e5586fa4cee91a02193f020e86658984798`; see `phone-candidate45zv-models.json`.
- Added hash-checked native replay input selection and sparse tensor capture. Frozen original tracing was exact through the final convolution, then found a cube-rounding difference in `pow_224`. The host's trailing-singleton loop uses powf; ordered square*x matches only949/1280 intermediate values. Later mapped linear operations are exact with the same captured inputs. `replay21-{prefix,middle,sparse,projection}*`, `cube-analysis-retry.json` retain diagnosis.
- Reused the strict OpenCL reference's compensated cube rule for the explicitly traced NCHW singleton activation. Spatial NHWC ordered cubes remain unchanged. The first guard used input dimensions, which ncnn's in-place dispatch leaves zero; that failed probe is retained (`replay21-cube-untapped.json`, candidate45zw). Corrected guard uses output dimensions; regression checks include in-place dispatch.
- **PASS** corrected native untapped replay21: all553 outputs bit-exact (`replay21-cube-dispatch-untapped.json`). This separate provider ran with candidate45zv installed; full candidate45zx APK acceptance is recorded separately below.
- candidate45zw APK SHA256 is retained in `../../build/candidates/candidate45zw/artifacts.json`; archived, not installed.
- candidate45zx APK SHA256 `d78a0f110ab595a711a6b296cc1e6f202462a816427c7fdd9d25a5376d710c53`; test APK `96876c828001aafe773f8d00589528522dce16709372ab77e134117186de6a6f`. Reproducible patch/provider identities: `../../build/candidates/candidate45zx/artifacts.json`, `candidate45zx-source-identity.json`.
- Emulator-only fixture changes distinguish cold Python import and receiver readiness from actual frame delivery, keep producer alive through the receiver's bounded acquisition, and always join before releasing its publisher. Backpressure test allows cold software rendering while retaining frame-count, drop, ownership, ordering and teardown assertions. Runtime startup allows180s on emulator only (phone60s unchanged); readiness and all active-runtime watchdog/timing gates remain intact. Prior candidate45zu failures are retained, not reclassified.

### Installed candidate45zx model gate

**PASS driver accuracy:** all159 comparisons (3 initial,96 seed-sequence,60 replay) pass against the unchanged originals (`phone-candidate45zx-models.json`). No unexpected non-finite values; assignment records reject CPU fallback. Candidate45zx is still diagnostic and performance is not run.

**BLOCKED combined795 gate:** road ncnn lacks a conversion verified against the original graph. Unsupported existing road conversion is retained in `road-existing-conversion-audit.json`. The validator rejects it explicitly, and accelerator timing is not advanced. The accuracy wrapper therefore correctly returns failure despite driver accuracy passing. Physical camera identity, calibration and Panda gates remain blocked independently.

## Final candidate45zx validation (2026-11-05)

| Gate | Result | Evidence / limitation |
|---|---|---|
| Focused host tests | PASS | 208 tests, `host-dispatch-full-tests.log`; focused lint passed, broader pre-existing lint failures retained |
| Driver original-reference accuracy | PASS | All159 initial/sequence/replay comparisons; unchanged tolerances, no CPU fallback |
| Combined795 model gate | BLOCKED | Road conversion unsupported/unverified; no accelerator performance qualification |
| Fold3 instrumentation | PASS | 28/28, `phone-candidate45zx-instrumentation.txt` |
| Fold3 steady and fault fixtures | PASS | Separate30-second injected full-chain runs; not live camera/model qualification |
| Emulator instrumentation | PASS | 28/28, `emulator-candidate45zx-instrumentation.txt` |
| Emulator steady and fault fixtures | FAILED readiness / NOT RUN | Both180-second startup gates failed before fixtures; supervised processes alive, Raylib insufficient frames. Isolation passed. Logs retained, including runtime logcat. No watchdog relaxation |
| Controls work | PASS short fixture | p99 2.249322ms, mean100.005Hz, work/scheduled misses0, skipped0 |
| Controls complete cycle | FAILED | p99 12.299689ms;1217 raw cycle deadline misses retained |
| Sustained combined workload | BLOCKED | Accuracy/freshness/timing gates prevent advancing to2/30-minute qualification |
| Physical qualification | BLOCKED | Rear sensor identity, mounted calibration, disconnected Panda and vehicle remain unverified |

The same controls fixture hash is recorded in `controls-comparison-zx.json`. Baseline45zq versus45zx: work p99 2.692917→2.249322ms; complete cycle13.561250→12.299689ms; scheduler delay3.058380→1.276092ms; release-to-completion5.821448→3.996814ms. These are short matched fixtures with no concurrent host build or numerical workload, not sustained performance acceptance. Candidate44's earlier16.61ms complete-cycle result remains historical, not a matched final comparison.

APK SHA256 on both installed devices: `d78a0f110ab595a711a6b296cc1e6f202462a816427c7fdd9d25a5376d710c53`. Installed verification receipts are `phone-candidate45zx-installed.json` and `emulator-candidate45zx-installed.json`. TestAPK SHA256 `96876c828001aafe773f8d00589528522dce16709372ab77e134117186de6a6f`. TX remains compiled out and engagement disabled. Cooling optimization was not performed; safeguards and30-second cool/clean isolation were retained.

Reproduction wrappers (archive uses exclusive result names: choose a new report directory/labels before rerunning):

```sh
.venv/bin/python android/reports/final-head-20261104/run_accuracy_zx.py
.venv/bin/python android/reports/final-head-20261104/run_devices_zx.py
```

Each wrapper records its exact underlying commands and artifacts. The accuracy wrapper intentionally exits nonzero while the combined model gate is blocked. The device wrapper checks isolation for each attempt; its own exit status is not a substitute for each validation receipt. Earlier failed candidates, failed probes and emulator attempts remain preserved.

Final cleanup verified: both devices have no app/test processes, no owned camera and no experiment forwards; NORMAL thermal status continuously exceeded30seconds. Dedicated emulator was shut down and ADB lists only the Fold3. Receipts: `phone-candidate45zx-final-cleanup.json`, `emulator-candidate45zx-final-cleanup.json`.
