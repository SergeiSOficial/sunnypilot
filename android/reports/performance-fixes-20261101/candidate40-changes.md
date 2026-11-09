# Candidate40 implementation

Branch `codex/fold3-android-port`; app SHA-256
`bd42b65a3538b8ad73834e90826eb10a98d8afeacbe9d5aae5877d43a25e30ec`.
TX remains compiled out and physical engagement disabled.

## QNN diagnostics and accuracy

The Android ORT binding accepts provider options and exposes profiling results
and assigned node names, domains and operators. It preserves the selected backend
and rejects attempts to override it through the diagnostic option dictionary.
The QNN trace path uploads separately hashed graphs and records tensor type,
shape, declared layout, first tolerance violation, unexpected non-finite values,
final outputs and provider assignment. Raw traces and failing tensor arrays are
retained. Exposing an intermediate can change fusion or partitioning, so a tap
passing is never sufficient for qualification.

Portable host-precision exports preserve the original half-precision Gemm
boundaries. The image-layout export moves lossless byte rearrangement into FP16,
which represents every byte exactly, and folds the history image's trailing
spatial axes so QNN's Slice/Concat operations stay within four dimensions. The
external image interface remains uint8. Host checks require agreement with the
original final outputs. These exports remove the remaining CPU partitions from
the tested QNN graph, but QNN still fails the numerical gate. They are diagnostic
assets and are not selected for live perception. HTP initialization remains
deferred.

Final candidate40 tests also run derived graphs with no intermediate outputs.
QNN numerical accuracy still fails, and the untapped road graph produces 998
unexpected non-finite output elements that do not occur in the tapped graph.
This full-graph failure remains separate from the exact OpenCL reference gate.

## Accurate OpenCL

The event profiler records bounded OpenCL kernel/transfer events without waiting
after each kernel. It drains submitted work before reading profiling timestamps,
checks ordering, restores hooks and releases owned events on success or error.
Compilation, enqueue cost, queue delay, GPU execution and readback are separate.
Warm complete calls are compared without serialized tensor checkpoints.

Measured ordered convolutions dominated the accurate path. Candidate38 introduced scheduling of
up to 128 independent output lanes per workgroup while preserving each output's
accumulation and rounding order. Activation/division kernels retain 32 lanes.
Candidate39's wider convolution and activation/division workgroups regressed and
were reverted. Original models, goldens and `rtol=1e-3, atol=1e-4` are unchanged.

Qualification records bind the actual APK, models, references, implementation,
provider library and device. CPU fallback and diagnostic-only results cannot
qualify live inference. Accuracy, full model coverage and timing are independent
requirements. No backend is qualified by this delivery.

## Camera and controls measurements

Camera reports now distinguish raw sensor capture to capture-result delivery
from texture acquisition, GPU conversion, readback and publication. Raw sensor,
elapsed-realtime and monotonic clock samples remain available. Invalid clocks
are rejected rather than replaced with arrival times. Existing owned buffers,
newest-frame selection, bounded queues, GLES3 conversion and PBO/fence readback
remain in use. The four supported profiles are compared with the same continuous
readback workload. No rear sensor identity or mounted geometry is inferred from
transport success.

Android controls replace the fixed 15 ms poll with a wait bounded by the remaining
10 ms cycle budget. An early-returning poll waits only for the residual time
to that same absolute deadline; this prevents the candidate38 overspeed seen
when input was immediately available. Polling and residual wait share one
deadline, and both remain visible in cycle timing. Other platforms keep the
existing 15 ms behavior. A zero IPC timestamp is reported as unavailable (-1),
rather than converted into an uptime-sized age. Reports retain complete-cycle
misses and additionally separate the wait budget, cycle interval, scheduler lag,
IPC message age, input handling, computation, publishing and reporting. Control
calculations, cancellation, stale-input handling and watchdogs are unchanged.

## Validation discipline

Cooling and temperature optimization are deferred at the user's request.
Existing thermal protection and experiment cleanup remain active.

Experiments stop the application and test processes, check camera release and
provider stability, remove experiment-created forwarding and verify the APK
before and after the child. A new experiment requires 30 continuous NORMAL
thermal seconds and phone skin temperature at or below 32°C. MODERATE thermal
pause and critical abort remain active. Phone performance runs keep the emulator
stopped. The final command inventories retain individual validator and isolation
results, including failures. Longer combined workloads require the short
correctness, input and timing gates first.
