# Candidate45zb convolution optimization

This candidate builds on the complete-model accuracy repair in
[candidate45za](candidate45za-implementation.md). Vehicle TX remains compiled
out, physical engagement is disabled, and original models, goldens and
`rtol=1e-3`, `atol=1e-4` are unchanged. Cooling optimization is deferred.

## Measured operator cost

The Fold 3 OpenCL device reports `CL_DEVICE_SINGLE_FP_CONFIG=22`, without
`CL_FP_FMA`. The existing correctly rounded, ordered `fma` convolution is
expensive on this driver. The retained
[production-kernel probe](fold45za-conv-production-probe.json) compares that
strict implementation with a vector-load implementation using four sequential
`mad` updates in the same kernel-row, kernel-column and input-channel order.
It uses the actual production kernel source, with the new macro off and on;
no CPU fallback is allowed. This diagnostic ran while candidate45za was
installed; its separate binary and kernel-source hashes are recorded. It is
not a full-model performance result for either APK.

All 36 synthetic-operand cases pass the original tolerance, with no non-finite
values and maximum absolute difference `2.09808e-05`. Coverage includes three
actual model convolution geometries, three seeds, both bias orders, odd channel
tails, batches, stride, dilation, padding and ReLU. The three large geometries
have median GPU event times of approximately 259 to 5.15 ms, 228 to 4.98 ms and
60.8 to 1.36 ms. Events are collected after batch submission; there is no
per-kernel synchronization. Compilation and host transfers are separate.
These are five-sample operator medians, not inference p99 or sustained results.

Earlier probes remain retained. Removing volatile storage or adding vector
loads alone gave smaller improvements. An initial vector probe failed to
compile; its corrected version regressed one static pointwise geometry and
was rejected. The accepted probe and its experiment wrapper preserve those
failures rather than replacing them with the later result.

## Bounded production change

The reproducible `mnn-half-cast.patch` adds `FOLD3_VECTOR_MAD_CONV` only for
regular, original-FP32 convolution operations already routed through the
host-order buffer kernel. Depthwise convolution and the separately qualified
paired half-product operation keep their existing arithmetic. Explicit FP16
Cast boundaries, attention reductions, recurrent state and external model
semantics are unchanged. The strict FMA implementation remains available when
the macro is omitted, and candidate45za retains the exact prior provider
libraries and patch.

No global fast-math flag, precision conversion, CPU partition or relaxed
tolerance is introduced. The source is pinned to MNN commit
`7e3c6fb5225d3ccf22807ee641332669df34650d`. The final patch dry-run and native
build pass. A malformed intermediate patch-generation attempt is retained in
`candidate45zb-mnn-build-patch-generation-failed.log`; it was corrected before
the successful build. All 795 host graph-conversion comparisons, 133 focused
host tests and 1,921 Hyundai/Panda/simulator tests (202 skipped) pass.

## Acceptance and provenance

**Rejected:** the untapped phone graphs fail the initial three-seed checks for
both models. Nine of 15 output comparisons pass; six fail. Road output failure
counts are 160, 324 and 302 elements, and driver counts are 8, 12 and 4.
The [rejection record](candidate45zb-rejection.json) retains these results and
their APK identity. Recurrent/replay suites, inference timing and combined
workloads were not advanced. Both initial experiments exited cleanly. The APK
was installed and hash verified on the emulator but was not validated there
after the numerical rejection. Candidate45zc restores correctly rounded FMA;
it requires separate acceptance evidence.

Operator agreement cannot establish full-model agreement: a small convolution
difference can cross an intentional FP16 rounding boundary. Candidate45zb must
repeat all 795 complete, untapped phone comparisons, including recurrent
sequences and the pinned camera replay, before inference timing. Its APK,
packaged stripped provider libraries, source patch and converted graphs have
separate immutable hashes. Candidate45za's proof cannot qualify candidate45zb.

Controls scheduling and reporting are unchanged from candidate45za. The same
fixture is repeated, preserving complete-cycle latency, raw intervals and all
legacy deadline counters alongside the newer release-to-completion metrics.
Emulator results establish correctness and recovery only. Physical accelerator
timing and every subsequent workload stage retain their separate gates.
