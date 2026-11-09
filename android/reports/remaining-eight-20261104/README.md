# Full-model and controls timing repair — 2026-11-04

Vehicle TX remains compiled out and engagement disabled. Fold 3 and Panda are disconnected; physical performance, mounted calibration and independent rear-camera identity remain blocked. Cooling optimization is deferred; thermal safeguards and experiment cleanup remain enforced.

## Changes

- Added same-input ONNX section comparisons for actual captured ncnn Vulkan Conv/linear tensors, bound to original graph and tensor hashes. These are diagnostic, not full-model qualification.
- Preserved the FP16 half-GEMV repairs. Added sequential FP32 FMA with bias last for the driver projection convolution, supporting Vulkan's internal four-channel packing. Extended the same proven arithmetic to the two immediately preceding 1x1 projection convolutions. Original models, goldens and tolerances are unchanged.
- Added controls thread CPU/off-CPU diagnostics. Wall deadlines, complete cycles, raw intervals, skipped slots and deadline misses remain unchanged. Diagnostics alone do not clear the timing blocker.

## Preserved intermediate failures

| Artifact | Seed 42 untapped driver mismatches | Result |
| --- | ---: | --- |
| candidate45zl | 8 / 553 | Previous baseline; unqualified |
| candidate45zm | 8 / 553 | Scalar-only shader guard did not cover Vulkan's channel packing; rejected |
| candidate45zn | 1 / 553 | Packed projection kernel active; still unqualified |
| candidate45zo | 1 / 553 | Three projection convolutions use ordered accumulation; still unqualified |

candidate45zm APK SHA256: `d97f5adb31534d60fd4bd4e7af73b8b8a7605731e229068af4eb3c6884c92e5e`.
candidate45zn APK SHA256: `90ae7772fd7640cf079d2edece7f570a583b6595e14f1e49e33e02ca90394db7`.
Both were installed-hash verified on the dedicated ARM64 Android 15 emulator. Software Vulkan is correctness evidence only.

The candidate45zn GPU trace confirms Conv99 and linear100 are exact on their respective captured inputs. Conv97 and Conv98 remain locally within tolerance but differ in FP32 rounding. The ordered host implementations of Conv97, Conv98 and Conv99 are exact for captured inputs and seeds 42, 7 and 123 (12 section checks). Full-model results govern acceptance, never these sections.

## Host/build evidence

- `host-suite.log`: 190 unittest checks passed, including scheduling, IPC, supervision, preprocessing, Hyundai and Panda restrictions.
- `projection-prefix-host.json`: 12/12 exact original-section checks.
- `prefix-shader-check.log`: actual shader compilation across FP32, FP16 storage and FP16 arithmetic.
- Build, conversion and APK archive logs preserve each candidate. Conversion retains all 159 original external input/output checks.
- `packed-heads-trace-seed42-same-input.json` records actual GPU local arithmetic; `projection-heads-trace-seed42-same-input.json` preserves the ineffective scalar-guard result.

One trace start was correctly rejected while ADB still listed the exiting emulator (`packed-trace-start-rejected.log`). The subsequent run began after it disappeared. Every completed experiment retains its cleanup receipt; no simultaneous camera/model/timing experiments were run.

Final candidate and emulator/controls results are recorded in `validation-summary.json` when complete. No accelerator qualification or physical readiness is implied by emulator fixtures.

## Final artifact and controls comparison

Candidate45zo APK SHA256: `d3732836486d1c1518b8b05462ce45e520cd06a7d93ae8968f62ebedd2161c45`. The installable APK is in `android/build/candidates/candidate45zo/app-debug.apk`; `artifacts.json` binds models, providers and patches. Emulator installed-hash verification passed. Physical Fold 3 installed-hash verification is blocked while disconnected.

Untapped seed 42 still fails 1/553 comparisons (max absolute error 0.001953125), all finite with no CPU fallback. The failing output changed from candidate45zn, demonstrating remaining upstream numerical sensitivity. Seeds 7/123 and the complete 795-suite were not advanced after seed 42 failed. No model is promoted. The candidate45zo full GPU prefix has not been locally traced; only the candidate45zn Conv99 GPU section and the 12 host sections are proven exact.

Matched 30-second synthetic controls fixtures (same fixture hash; sequential host-GPU emulator runs, no competing host builds/tests):

| Metric | candidate45zl | candidate45zo |
| --- | ---: | ---: |
| Work p99 ms | 1.637 | 1.310 |
| Complete cycle p99 ms | 11.399 | 11.192 |
| Release-to-completion p99 ms | 2.400 | 2.223 |
| Scheduled deadline misses | 1 | 1 |
| Skipped slots | 4 | 0 |

This single pair does not establish a scheduling improvement: scheduling behavior is unchanged, and host/emulator variation remains. Complete-cycle timing remains failed. Candidate45zo reports active thread CPU p99 1.374 ms, active off-CPU p99 0.716 ms, scheduler delay p99 0.891 ms and reporting p99 0.131 ms. Percentiles are separate distributions and must not be summed. All 1334 raw complete-cycle misses remain recorded.

A separate repeat in the final validation inventory measured work p99 1.093 ms and complete-cycle p99 11.396 ms, with zero scheduled misses/skipped slots. It is retained alongside the A/B pair, not substituted for the less favorable result.

## Reproduction

`reproduce.sh` records the pinned host compiler, model preparation, Gradle flags and sequential device runners. Choose a new archive name for any rebuild; completed inventories intentionally skip existing rows. The full-model runner stops advancing seeds on the first accuracy failure. Device wrappers stop app/native probes, verify camera release, clear only experiment forwarding, and enforce the established 30-second NORMAL/skin-temperature baseline before and after experiments. Runners shut down the dedicated emulator when complete.

## Final emulator validation

28/28 instrumentation checks and 9/10 experiment groups passed. Process-chain steady/fault fixtures, lifecycle recovery, ADB faults, IPC, MetaDrive cancellation/stale input and thermal handling passed. Camera timing failed: all 290 capture-result timestamps were in the future (`delivery_future_timestamps=290`); invalid capture times remain rejected. All ten isolation receipts passed. Final cleanup verified no owned camera, no experiment processes/forwarding, normal cooldown, and shut down the dedicated emulator. ADB lists no connected device afterward.

Full-model accuracy and complete-cycle timing are **FAILED**, not fixed or qualified. The remaining output mismatch requires tracing earlier convolution/activation propagation before another arithmetic change. Physical Fold 3 performance, final installed hash and sustained-load qualification are **BLOCKED** while disconnected. No 2-minute or 30-minute accelerator workload was advanced past these failed gates.
