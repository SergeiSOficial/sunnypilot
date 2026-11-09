# Fold 3 reconnection and Vulkan arithmetic diagnosis — 2026-11-04

Candidate45zp preserves candidates35/40/44/45zo. TX remains compiled out and engagement disabled. Cooling optimization remains deferred; thermal and cleanup gates remain enforced.

## Verified change

On the physical Adreno 660, all captured outputs of the three marked projection convolutions matched separately rounded multiply/add exactly despite the shader requesting GLSL `fma`. The original host reference instead uses fused accumulation. `projection-arithmetic.json` binds the diagnostic to actual captured inputs; fused/separate comparison counts are retained.

The reproducible ncnn patch now uses compensated product/addition residuals only in the existing diagnostic branch for these three convolutions. Actual GPU same-input comparisons after rebuilding match the original reference exactly for **12,288 + 92,160 + 1,280 = 106,240 elements**, versus 3,258 + 33,904 + 349 before. This establishes the targeted arithmetic repair on these captures; it does not qualify all FP32 values, the complete graph or inference performance.

`compare_ncnn_same_input.py` now also isolates original Tanh, ReduceMean and Cast nodes, retaining original input dtypes/rounding, captured-input hashes and graph hashes. Two focused regression tests cover these semantics and capture-corruption rejection.

## Remaining accuracy failure

Seed42 physical Vulkan full output still fails **7/553 elements**, maximum absolute error 0.00390625, before and after the targeted repair. Earlier emulator candidate45zo had one failure; the device results are distinct and cannot be substituted. Earlier convolutions and nonlinear operations retain small numerical differences. Seeds7/123, recurrence, replay, the full 795-comparison qualification and accelerator timing must not advance on this failure. Original graphs, goldens and rtol=1e-3/atol=1e-4 are unchanged. No CPU fallback is allowed or selected.

The standalone traces use separately pushed provider/probe binaries and include their hashes. Their installed app at capture time was candidate45zg/zh (same APK hash); they are not APK execution qualification. Candidate45zp packages the repaired implementation and has separately verified installed hashes in the device validation receipts.

## Host and device evidence

- Host launch/import failure recovered without any security configuration change; Fold 3 ADB serial RFCR81Y1TBJ is available.
- Complete final host suite: **192 PASS** with pinned Xcode26.5 SDK (`host-tests-final.log`, 22.618 seconds), including the two new same-input regression tests. The initial default-SDK run failed two compiler checks and remains retained.
- Changed Python files: Ruff PASS. Working-tree whitespace check: PASS.
- Native ncnn build and three shader compile variants: PASS. Gradle build: PASS (JVM unit task up-to-date).
- Candidate45zp phone and emulator instrumentation: **28/28 PASS each**; installed hashes, artifact stability and isolation PASS.
- Both devices pass 30-second steady-input and cancellation/stale-input fixtures. These use synthetic perception/calibration, not camera/model-qualified inputs. Every experiment's process/camera/forwarding cleanup and cooldown receipt passes. The task-owned emulator is shut down after testing.
- Trace-free seed42 checks fail **7/553 on the Fold 3** and **1/553 on the software-Vulkan emulator**, with no unexpected non-finite outputs (`phone-untapped45zp.json`, `emulator-untapped45zp.json`). The targeted repair does not establish complete untapped-graph agreement.

Machine-readable passed/failed/blocked results are in `validation-summary.json`. Earlier failed runs are retained; the summary distinguishes functional fixture passes from numerical/timing qualification failures.

## Matched controls measurements

Same fixture SHA-256 `b5ea99327f0ffc121e5556d29b32d4ce8e6e855d14518f46a3f060ea38721f99`, 30 seconds, no concurrent host builds/numerical tests. These are full process-chain fixtures with synthetic inputs, not combined accelerated perception workloads. See `controls-comparison.json` for every stage and deadline counter.

| Experiment | Work p99 ms | Complete cycle p99 ms | Scheduled misses | Skipped slots | Raw cycle misses |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fold 3 candidate45zo steady | 2.576 | 13.629 | 0 | 0 | 1272 |
| Fold 3 candidate45zp steady | 2.918 | 13.605 | 2 | 0 | 1293 |
| Fold 3 candidate45zp faults | 2.577 | 13.816 | 0 | 0 | 1285 |
| Emulator candidate45zp steady | 4.085 | 13.858 | 12 | 30 | 1261 |
| Emulator candidate45zp faults | 3.897 | 13.102 | 5 | 1 | 1286 |

**Controls timing: FAILED.** Work meets the 10 ms p99 target in these short fixtures, but complete-cycle jitter and deadline misses remain. Candidate45zp changes GPU arithmetic, not controls scheduling; timing variation is not evidence of an optimization. Raw cycle definitions/counters are preserved. Fold steady mean frequency is 99.992 Hz; emulator steady is 98.894 Hz. Emulator timing cannot qualify phone accelerator performance. No accuracy/timing promotion or longer combined-load progression follows these failures.

Candidate45zp APK SHA-256: `863221c406c13821fd017751f023febab3f460346e97f3870242e2ebca48a5d0`. Both installed APKs match. APK, implementation, provider libraries and graph hashes are retained in `android/build/candidates/candidate45zp/artifacts.json`.

Mounted calibration, independent rear sensor identity, Panda firmware/passive CAN/harness and physical vehicle qualification remain blocked. No long combined accelerator workload is authorized by a failing numerical gate.

## Reproduction

Run commands from `/Users/sergios/sunnypilot`; preserve existing reports and choose new output paths when repeating probes:

```sh
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
export CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
.venv/bin/python -m unittest discover -s android/tests
.venv/bin/python android/scripts/build_ncnn.py
.venv/bin/python android/scripts/probe_ncnn.py --serial RFCR81Y1TBJ --adb /Users/sergios/Library/Android/sdk/platform-tools/adb --seed 42 --trace-count 64 --trace-skip 400 --report /tmp/NEW-fold3-trace.json
.venv/bin/python android/scripts/compare_ncnn_trace.py /tmp/NEW-fold3-trace.json
.venv/bin/python android/scripts/compare_ncnn_same_input.py /tmp/NEW-fold3-trace.json
```

`run_devices.py` records the matched phone baseline, candidate45zp phone/emulator fixtures, stable APK hashes and per-experiment cleanup. It refuses an existing emulator and existing experiment receipts; use a new report directory for repeats. `run_untapped.py` repeats seed42 with tensor tracing disabled on the physical GPU and software-Vulkan emulator, retaining separate provider/probe hashes; these standalone diagnostics do not qualify APK runtime execution. Experiments run sequentially; host builds/numerical tests do not overlap controls timing. APK: `android/build/candidates/candidate45zp/app-debug.apk`.
