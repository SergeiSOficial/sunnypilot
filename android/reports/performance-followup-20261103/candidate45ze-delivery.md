# Candidate45ze delivery

Branch: `codex/fold3-android-port`. Work continued in this chat. Vehicle TX remains compiled out; physical engagement remains disabled. Cooling optimization is deferred and thermal safeguards remain enforced.

## Artifact and implementation

[Installable APK](../../build/candidates/candidate45ze/app-debug.apk), SHA-256:
`54a613533c10795ab754e2737d5bf2facfd387dfb2bd58ef43017260dd063a3f`.
Both app and test APK installed hashes match the immutable archive on the Fold 3 and dedicated ARM64 Android15 emulator. The archive retains actual libraries, model graphs, conversion manifest and reproducible patch identities.

The MNN patch accelerates regular vector convolution only when both finite operands are exactly representable as binary16 and the accumulator is finite and normal or zero. The product is then exact in binary32, allowing separately rounded multiplication and addition while preserving the original accumulation order. Other inputs retain strict FMA. Depthwise, scalar and FDOT2 paths retain their previous arithmetic. Original model hashes, goldens and tolerances remain unchanged.

Added separate GPU-event profiling builds, a shell-only native probe and artifact-bound runner. The parser rejects incomplete event coverage, changed kernel order/count and inconsistent native totals. Profiling libraries are separate from the non-profiling APK libraries.

## Results

| Gate | Result | Evidence |
|---|---|---|
| Android host checks | PASS: 182 tests | `candidate45ze-host.log` |
| Hyundai safety and simulator protocol | PASS: 1921 run, 202 skipped | `candidate45ze-hyundai-safety-sim.log` |
| Guarded convolution probes | PASS: 144 synthetic cases, bit-exact | Both guarded probe reports and isolation records |
| Host conversion agreement | PASS: 795 comparisons | `candidate45ze-host-conversion.json` |
| Actual Fold 3 model agreement | PASS: 795/795, no fallback/non-finite failures | `candidate45ze-accuracy.json` |
| Fold 3 functional tests | PASS: 6 groups, including 28 instrumentation tests | `fold-candidate45ze-validation-driver.json` |
| Emulator retry | 9/10 groups PASS; native camera timing FAIL | `emulator-candidate45ze-retry1-validation-driver.json` |
| Live perception and vehicle qualification | BLOCKED | Timing and physical gates below |

Actual-device accuracy covers 159 driver and 636 road comparisons, all three seeds, recurrent sequences and the pinned 60-frame replay. `rtol=1e-3`, `atol=1e-4`. Neither model is promoted for live perception on accuracy alone.

## Matched performance

| Measurement | Candidate45zd | Candidate45ze | Gate |
|---|---:|---:|---|
| Driver inference p99, 100 samples | 1276.617 ms | 1058.942 ms | FAIL: all 100 exceed 50 ms |
| Road inference p99, 100 samples | 2261.544 ms | No completed p99 | BLOCKED: MODERATE thermal safeguard stopped the run |
| Controls work p99 | 2.448750 ms | 2.668698 ms | PASS: below 10 ms |
| Controls complete-cycle p99 | 14.000989 ms | 13.549531 ms | FAIL: legacy cycle timing remains open |

Driver p99 improved about 17%; this is insufficient for 20 Hz inference. No road p99 is inferred from its last-call stages or a partial sample set. Road timing cleanup and artifact isolation passed despite the blocked benchmark.

The matched controls fixture reports 2661 samples, 100.003 Hz, UI 29.693 fps, 1214 legacy cycle misses, zero work misses, zero scheduled-deadline misses and zero skipped slots. Release-to-completion p99 is 5.914315 ms; reporting p99 is 0.356927 ms. These schema-2 metrics supplement unchanged legacy counters. This repeat does not establish a controls root-cause fix. Earlier candidate44 results remain preserved; CPU-placement and environment differences limit causal comparisons.

Diagnostic profiling shows driver stem convolution median 211.288→94.146 ms. Road pointwise convolutions remain about 323 and 318 ms in aggregate; unary aggregates remain about 187 ms driver and 197 ms road. These all-zero, three-run event profiles identify remaining costs, not ordinary latency p99. See `candidate45ze-profile-summary.json`.

## Emulator failures and recovery

The first SwiftShader/Vulkan attempt passed instrumentation, then failed two chain startups with Raylib blocked in `end_drawing`. The inventory was interrupted before lifecycle completion; later pending rows are explicitly blocked. Its reports, native stack and manual cleanup record are retained.

Rebooting the same AVD with host OpenGL and guest Vulkan disabled resolved the observed startup stall without changing the APK. The fresh retry passes instrumentation, steady/fault chain fixtures, lifecycle, ADB recovery, IPC, MetaDrive cancellation, MetaDrive stale inputs and thermal handling.

Native camera timing still fails: all 289 delivered frames carry future timestamps, approximately 30 ms ahead, despite declaring REALTIME. The validator rejects these capture times. No arrival-time substitution or freshness-limit relaxation is used. Injected fixture success cannot qualify native camera timing.

## Remaining gates and cleanup

The 30-second, two-minute and 30-minute combined visual workloads remain blocked by inference performance, complete-cycle controls timing and physical camera gates. Independent rear-sensor proof, mounted calibration and real Panda/vehicle checks remain blocked: lens-cover trials are deferred, the phone is unmounted and the Panda disconnected. Simulator results cannot establish road readiness.

Processes are stopped between experiments; camera ownership and forwarding are checked. Final cleanup records verify at least 30 continuous NORMAL seconds and the established phone skin-temperature limit. The emulator is shut down after its final cleanup. Cooling configuration is unchanged.

[Reproducible commands](candidate45ze-reproduce.md), [artifact-bound summary](candidate45ze-validation.json), [full accuracy evidence](candidate45ze-accuracy.json). Earlier candidate archives and failures remain preserved.
