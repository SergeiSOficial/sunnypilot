# Fold 3 remaining-seven recovery — 2026-11-04

**FAILED accuracy and timing acceptance; not qualified. TX remains compiled out and physical engagement disabled.**

**The first restoration check (`final-untapped-device.json`) is invalid as restored-source evidence**: its binary retained the rejected GEMM shader through stale generated objects. Its 7-mismatch failure remains archived. A source timestamp/cache invalidation bug was found and fixed in `build_ncnn.py`; `source-cache-test.log` proves removed patches refresh source timestamps without weakening archive path validation. Fresh full rebuild and replacement acceptance check are recorded separately. **Fresh-cache untapped seed42: 1/553 mismatch**, output26 -1.046875 versus -1.048828125, max absolute error0.001953125, all finite (`fresh-cache-untapped-device.json`). This confirms a reduction from7 to1 on the first full model, but still fails original tolerance. No completed full-model accuracy fix or accelerator speedup is claimed.

Candidate45zq APK SHA256: `db2fc305eaaedb99b87a746fdad7816ab0d07206f1d2f641b8f5cc0c3a4085b1`.
Installed hash verified on Fold 3 RFCR81Y1TBJ and ARM64 API35 emulator-5560. See
`phone-candidate45zq-installed.json`, `emulator-candidate45zq-installed.json`, and
`../../build/candidates/candidate45zq/artifacts.json`. Candidate45zp is retained as baseline.

## Implementation and numerical evidence

- Extended ordered FP32 pointwise convolution to all FP32 activation 0/1 configurations, excluding FP16/BF16. Compensated FMA preserves the reference arithmetic on Adreno660. Actual same-input Conv93–96: 122880/122880 values exact. Trace-bearing seed42 mismatches reduced from 7 to 1; final untapped seed42 still fails with 7 mismatches.
- Corrected Tanh polynomial/residual FMAs using compensated arithmetic. Actual same-input Tanh66/67/68: 49152 + 92160 + 1280 values exact. This still leaves one full-model mismatch (its location changes with later precision corrections).
- Added ordered FP32 depthwise accumulation and compensated spatial-mean division. Original models/goldens/tolerances unchanged (`rtol=1e-3`, `atol=1e-4`). No CPU fallback or backend promotion.
- Probed spatial bias-first arithmetic against original captured sections. Direct-convolution shader diagnostic is bypassed by the selected GEMM path. The separate GEMM attempt initially failed shader compilation; braces fixed the macro expansion, but its complete output regressed to 7 mismatches. **Rejected and removed from active patch**, preserved in `rejected-spatial-gemm.patch` and `spatial-gemm*-device.json`.
- Active patch restored to candidate45zq identity `aa40051323448315def85ed7b3be5a1e6608226b5d4d4225ca151cc0d5c8d0e4`, rebuilt reproducibly.
- Latest trace-bearing retained model result: one mismatch of 553, output26: -1.046875 versus -1.048828125. Maximum absolute error 0.001953125; no unexpected non-finite outputs. See `spatial-pack1-device.json`. Traced diagnostic results do not qualify the untapped model. First trace-free restoration check: `final-untapped-device.json`, **invalid restored-source identity** due to stale generated shader objects; see fresh-cache check below.
- The full 795-comparison suite is **BLOCKED**, not passed: the first original-reference gate still fails. Recurrent/replay, accelerator performance and 2-/30-minute combined runs cannot advance.

## Controls: no timing qualification

Removed a duplicate per-value scan in the binary report writer; the manager continues to reject every non-finite decoded value. Added a regression proving fault values remain intact in the handoff. Absolute 10ms pacing, raw cycle metrics, missed-slot/deadline reporting, watchdogs, cancellation and stale-input handling remain intact. This change did **not** demonstrate a timing improvement.

Matched 30-second steady synthetic full-chain fixture, sequential, no builds/numerical host tests during timing:

| Fold 3 result | candidate45zp | candidate45zq |
|---|---:|---:|
| Controls mean Hz | 99.996 | 99.999 |
| Work p99 ms | 2.575 | 2.842 |
| Complete-cycle p99 ms | 14.102 | 14.597 |
| Reporting p99 ms | 0.356 | 0.357 |
| Raw cycle deadline misses | 1280 | 1316 |
| Scheduled release/completion deadline misses | 1 | 0 |

**Timing FAILED.** The new run is not a speedup; different metric definitions do not clear raw-cycle failure. Both functional fixtures passed with TX false. Baseline scheduler trace (diagnostic overhead; not performance comparison) showed runnable-to-running delay p99 1.854ms, maximum7.068ms (`controls-runqueue.json`). Adding the prime core regressed timing and restarted workers; rejected. No GIL/priority/security changes or real-time scheduling bypasses were added.

## Validation results

| Check | Result | Evidence |
|---|---|---|
| Host tests | PASSED, 195 | `host-source-cache-tests.log` |
| Default SDK host compile | FAILED, retained | `host-final-tests.log`: incompatible macOS27.2 TAPI architecture; use Xcode26.5 SDK |
| Gradle unit tests and debug/test APK build | PASSED | `gradle-final-spatial.log` |
| Fold instrumentation | PASSED, 28/28 | `phone-candidate45zq-instrumentation.txt` |
| Fold steady full chain | PASSED functional; FAILED timing | `phone-candidate45zq-steady.json` |
| Fold cancellation/stale/fault full chain | PASSED | `phone-candidate45zq-faults.json` |
| Emulator instrumentation | FAILED, 27/28 | `emulator-candidate45zq-instrumentation.txt`: slowPublisher acquisition >30 frames/2.5s failed |
| Emulator steady / fault chain | FAILED startup, fixtures not exercised | respective `*-isolation.json`: Python services still importing at readiness timeout |
| Final APK installed hashes | PASSED both devices | `*-installed.json` |
| Experiment cleanup | PASSED all isolation receipts | `*-isolation.json`, `*-final-cleanup.json`; no owned camera/processes/forwards; >=30s NORMAL baseline |
| Physical vehicle / mounted calibration / independent rear sensor proof | BLOCKED | phone unmounted, Panda absent, lens-cover trials deferred |

Emulator failures are retained. macOS `syspolicyd` (~675% CPU) and `trustd` (~100%) were observed after these runs (`host-load-final.txt`), making host contention a material limitation. They were not disabled or changed. Emulator results establish no accelerator/road performance. No cooling optimization was attempted; thermal safeguards and cooldown procedures remain active. The task-owned emulator was stopped.

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
.venv/bin/python android/scripts/probe_ncnn.py --serial RFCR81Y1TBJ --adb "$HOME/Library/Android/sdk/platform-tools/adb" --seed 42 --trace-count 0 --report /tmp/fold3-unique-parity.json
```

`run_devices.py` records the exact sequential install/hash/fixture/emulator commands. It deliberately refuses overwriting report receipts; copy to a fresh report directory for a new run. All device experiments must retain stop/camera-release/forward cleanup and cooldown gates. Diagnostic probe/provider hashes are bound separately from installed APK hashes in each probe record.

## Build cache correction

Pinned tar extraction now omits archived mtimes so removed source patches invalidate Ninja generated-shader and object dependencies. Build identity includes the recipe hash and invalidates previous qualification. The stale first restoration failure is retained, not counted as the corrected binary. `fresh-native-build.log` records the full recompile; `fresh-cache-untapped-device.json` binds the fresh provider, source, runner and installed APK hashes separately.

Fresh-cache untapped check completed with one tolerance failure, artifact stable, scoped native processes/forwards removed and final >=30s NORMAL cooldown passed. The backend remains diagnostic and the full795 suite remains blocked.
