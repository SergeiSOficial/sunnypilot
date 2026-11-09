# Fold 3 NPU-first inference recovery

**Overall: BLOCKED.** Candidate44 executes small HTP and GPU workloads, but no
new engine passes the full-model qualification gates. Vehicle TX is compiled
out and engagement remains disabled. Cooling changes are deferred; Android
thermal pauses, cooldown and aborts remain enabled.

Branch: `codex/fold3-android-port`, inspected HEAD
`7631d5248cc9efd10b962107aadacbb908a9f14d`; changes are in the working tree.
Target: stock Android 15 SM-F926B, Snapdragon 888 / Adreno 660.

## Delivery and reproducibility

- [Candidate44 APK](../../build/candidates/candidate44/app-debug.apk),
  SHA-256 `f18262c93c2132b1e61a1746af6b2c55dcae8820e3d8a9db0dd84bc5472c6360`,
  1,920,624,443 bytes. This is a diagnostic debug build containing models,
  original-reference data and multiple accelerator engines.
- [Instrumentation APK](../../build/candidates/candidate44/app-debug-androidTest.apk),
  SHA-256 `27ce2124ae041ab24fa3f321cf9ee864d849230a7cf89758bff9dad54f7f16a7`.
- [Artifact and native-library hashes](candidate44-artifact.json),
  [commands](reproduce.md), [acceptance record](candidate44-acceptance.json),
  [comparison](candidate44-comparison.json) and
  [retained failures](candidate44-retained-failures.json).
- [Final delivery checks](candidate44-delivery-checks.json) verify archive and
  evidence hashes, report links and experiment shutdown. Acceptance stays blocked.
- Installed app and test hashes match the archive after testing on both the
  [Fold 3](candidate44-installed-fold-final.json) and
  [emulator](candidate44-installed-emulator-final.json).

ORT core 1.31.1, the QNN provider package 2.7.0 and QNN runtime 2.51.0 are
retained. Alternative engines are pinned to
[MNN 3.6.2](https://github.com/alibaba/MNN/releases/tag/3.6.2) and
[ncnn/pnnx 20261020](https://github.com/Tencent/ncnn/releases/tag/20261020).
Sources, tools and libraries have pinned download hashes; derived graphs and
weights have separate identities. No original model or golden was regenerated.

## HTP: executes quantized work, fails prerequisites

[Twelve probe results](candidate44-htp-probes.json),
[native validation log](candidate44-htp-probes-native.log) and
[QNN/ORT profile files](candidate44-qnn-profiles/manifest.json) distinguish
operator/precision validation from loader or DSP-access failure. FP16 Conv,
Gemm, activation and layout are rejected; a16w16 Conv/Gemm are rejected.
Native QNN validation reports code 3110 and ORT refuses CPU fallback.
No unsupported combination is promoted or retried as a CPU implementation.

The NPU does execute a16w8 Conv/Gemm and quantized activation/layout graphs:

| Probe | Original-reference result | QDQ CPU comparison | Decision |
| --- | --- | --- | --- |
| a16w8 Conv | 102 mismatches; max absolute error 0.0009580031 | 24 mismatches | FAILED |
| a16w8 Gemm | 8 mismatches; max absolute error 0.0004168861 | Passes tolerance | FAILED |
| a16w16/a16w8 activation | Passes tolerance | Passes tolerance | PASSED, probe only |
| a16w16/a16w8 layout | Exact | Exact | PASSED, probe only |

Each executed graph has one QNN partition and zero CPU computation assignments.
Calibration uses seeds 1000–1015; probe validation uses 2029. These are separate
from original validation seeds 42/7/123 and the pinned camera replay. Reports
retain tensor types/shapes, quantization nodes, graph hashes and actual operator
assignments. Passing the QDQ implementation alone cannot clear original-model
accuracy. Candidate42's earlier Gemm verdict is corrected in candidate44;
the original failed result is retained.

Full-model HTP quantization and performance are **BLOCKED** by these primitive
support/accuracy gates. This does not establish that every possible HTP graph
is impossible; it rejects the tested modes and transformations.

## Alternative GPU engines

All [eight small GPU probes](candidate44-gpu-probes.json) pass on the Fold 3
with actual no-CPU-fallback audits. Their [conversion manifest](candidate44-gpu-probes-manifest.json)
retains explicit types, shapes, batch contracts and separately hashed weights.
The largest of these tiny probes has MNN complete-call p99 1.68 ms; ncnn p99
is below 0.79 ms. These are not full-model timings or a comparison with the
accurate OpenCL baseline.

[MNN full models](candidate44-mnn.json) remain **BLOCKED**: the driver graph
selects CPU `InsertedPrecisionFreeCast_div:BinaryOp`; the road graph selects
CPU `next_state_img_q_raster_0:Raster`. The native audit rejects those executions.
Host-visible upload, compute and readback timing is available after successful
audits; serialized audit calls are excluded from subsequent timing samples.
These wall-clock stages do not establish exclusive hardware kernel time.

[ncnn full models](candidate44-ncnn.json): the driver graph loads with Vulkan
assignments, but all 553 outputs mismatch on each of seeds 42, 7 and 123.
Maximum absolute errors are 7.283203125, 7.0693359375 and 8.070068359375;
no unexpected non-finite values occur. Recurrent, replay and timing stages are
not advanced. The road conversion is **BLOCKED** before loading because pnnx
leaves `GatherND`, `ReduceMax`, `Tensor.reshape`, `Tensor.slice`,
`pnnx.Expression` and `torch.where`, and an unrepresentable rank-five image
state input. A successful converter exit is insufficient.

The implementation fixes tested integration faults: MNN's OpenCL library is
linked explicitly; ncnn's shader compiler is rebuilt against the app NDK;
native tensor copies respect padded channel/batch strides; converter batch
axes are recorded explicitly instead of inferred from the first dimension.
Small Gemm's matrix rows are no longer mistaken for independent batches.
MNN byte image transport is lossless, and only proven output-unreachable
`action_t` may be omitted. No CPU fallback is silently selected.

## Original numerical contract

Original hashes remain:

- Driver: `dee5a294e8afaacc9295ac5d100e00733ecac278e79264b96331e40a3ede1b04`.
- Road: `65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce`.

The [host export checks](candidate44-host-conversion.json) pass all **795**
original-reference comparisons for both the portable export and the separately
hashed MNN byte-transport rewrite: 159 driver and 636 road comparisons, including
three seeds, 32 recurrent frames per seed and the pinned 60-frame replay.
This host result does not qualify the converted device engines.
Tolerances remain `rtol=1e-3`, `atol=1e-4`.

A [failed diagnostic FP32/FP16-boundary rewrite](host-precision-seed42.json)
retains per-result half rounding but first diverges at Gemm:
[driver `node_linear`](host-precision-dmonitoring_model-trace.json),
max error 0.005859375, 274 elements; and
[road `model/p_node_linear_5`](host-precision-driving_supercombo-trace.json),
max error 0.03125, 24 elements. Per-result rounding does not reproduce the
original host's half-precision Gemm accumulation. This failed rewrite is
excluded from the APK. The original strict OpenCL implementation remains the
accurate reference; its candidate35/40 measurements are preserved.

Qualification is per model and binds the actual APK, original/derived models,
reference/sequence/replay manifest, implementation, native/provider libraries,
GPU driver and device fingerprint. CPU fallback, incomplete comparisons,
numerical failure, stale identities or missed timing gates cannot qualify live
perception. No new qualification record was produced.

## Validation and remaining gates

The build, Java unit task and **120 focused host tests** pass. The unchanged
Hyundai/Panda/protocol checks pass **3,940 tests, 356 skipped**. The initial
default macOS compiler failure and successful Xcode retry are retained in
[host evidence](host-checks.json).

All six [Fold 3 validation groups](fold-candidate44-validation-driver.json)
pass: 23 instrumentation tests, steady/fault process-chain fixtures, lifecycle
recovery, numerical/stale/cancel/thermal/USB rejection fixtures and IPC. These
use injected or diagnostic inputs and cannot qualify perception or a vehicle.
[Final phone cleanup](candidate44-fold-final-state.json) confirms no app/test
processes, no owned camera, no forwarding, NORMAL thermal status and 28.4 C skin.
All experiments separately retain isolation, installed hash and cleanup results.

The phone's [steady source fixture](fold-candidate44-chain-device-steady.json)
runs controls at 99.91 Hz with work p99 7.30 ms, complete-cycle p99 16.61 ms
and **1,350 complete-cycle deadline misses**; Raylib averages 29.68 fps.
The identical recorded fixture/pacing on candidate40 measured 8.83/18.29 ms
and 1,368 cycle misses. These are separate cooled runs with injected perception;
variance is not attributed to an accelerator improvement. Functional fixture
passes do not clear complete-cycle timing or sustained visual operation.

All eight emulator validation groups pass: six in the
[initial inventory](emulator-candidate44-validation-driver.json), plus
MetaDrive cancellation/stale scenarios in the
[retry inventory](emulator-candidate44-retry1-validation-driver.json).
Both devices pass **23 instrumentation tests**. The initial two MetaDrive
commands failed before simulation because a resolved base Python interpreter
lacked `capnp`; those failures remain. The corrected commands use the repository
virtual environment. MetaDrive reports zero longitudinal-active samples and
zero below-minimum-speed lateral-active samples in both retries.

The original 6 GB AVD could not install the large diagnostic APK, including
after removing only the port packages. Its
[installation failures](candidate44-emulator-install.json) are retained. The
[successful installation](candidate44-emulator-fresh-install.json) uses a new
dedicated Android 15 ARM64 AVD with a 12 GB data partition; the old AVD is
preserved. [Final emulator cleanup](candidate44-emulator-final-state.json)
confirms no app/test processes, camera ownership or forwarding; the emulator
is stopped after final hash verification.

Real emulator cameras and deterministic injected timestamps remain distinct;
candidate40's real-camera future-timestamp failure is retained and not cleared
by these fixtures. Emulated accelerator performance is never used to qualify
the Fold 3.

Accurate isolated full-model timing, 30-second combined operation, two minutes
and 30 minutes are **BLOCKED**; prerequisite accuracy/operator gates fail.
There is no new demonstrated 20 Hz inference, 100 Hz complete controls timing,
fresh-camera or watchdog qualification. Camera latency and controls contention
remain the recorded candidate40 failures, deferred until accurate inference
is selected. The candidate44 fixture retains complete-cycle timing failures
above. No temperature optimization or protection override was attempted.

The phone is unmounted and the Panda is disconnected. Independent rear-sensor
identity awaits the deferred lens-cover trials; mounted calibration, passive
vehicle fingerprint, firmware, harness and real vehicle behavior remain
separate **BLOCKED** gates. No simulator success clears those gates.
