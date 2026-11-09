# Candidate45za implementation

Vehicle transmission remains compiled out. Physical engagement is disabled.
Original models, goldens and `rtol=1e-3`, `atol=1e-4` are unchanged.

## Road attention reduction

The candidate45y full road graph first fails the seed42 recurrent sequence at
frame26. Actual candidate45z OpenCL traces show that both inputs to
`model/p_matmul_1` agree exactly with the original reference. One FP16 output at
`[0,4,8,27]` differs by `1.52587890625e-05`; the downstream projection amplifies
this to 74 output elements outside tolerance. See
[the captured-input reduction diagnosis](road45z-attention-matmul-reduction-frame26.json)
and [the original-host tail injection](road45z-cut-matmul1-seed42-frame26.json).

On the captured operands, paired FP32 product accumulation reproduces the
original host result. Serial FP32 accumulation reproduces the incorrect GPU
value. Candidate45za marks only statically verified batched FP16 MatMul nodes
and selects a GPU implementation that retains the paired rounding boundaries,
odd reduction tail, batch offsets and transpose semantics. The existing explicit
FP16 output Cast remains in the converted graph. FP32 and rank-two operations
retain their existing paths. No CPU partition is introduced.

The repair is reproducible through `prepare_accel_models.py` and
`patches/mnn-half-cast.patch`, against pinned MNN commit
`7e3c6fb5225d3ccf22807ee641332669df34650d`. Focused tests cover broadcast
geometry, preserved rounding, the captured odd-tail example and converter
markers. All 795 host graph-conversion comparisons pass. Complete untapped
phone comparisons remain a separate acceptance gate.

## Trace and recovery validation

Actual recurrent input archives retain their source hashes and frame provenance.
Trace validation now compares hashed uint8 image-history outputs using the
sequence manifest rather than attempting to load nonexistent golden arrays.
Diagnostic readbacks remain excluded from inference timing.

Thermal recovery validation now treats camera timestamps as belonging to the
current device boot. It requires frames after the current launch boundary and
distinct from the prior generation. It does not order persisted timestamps
across emulator reboots. Raw capture timestamps remain mandatory; future or
invalid capture times are rejected without substituting arrival time.

## Controls

Android controls pace once at the beginning of an absolute 10 ms slot, consume
inputs without blocking and skip expired slots without catch-up bursts. Other
platforms retain their input behavior. The controls thread hands off owned,
bounded timing batches; percentile work occurs in the manager process.
Heartbeat, reporting failures, cancellation, stale-input handling and watchdog
checks remain enforced.

The retained scheduling schema reports input handling, input age, scheduler
delay, computation, publishing, extensions, reporting and release-to-completion.
Raw intervals and the legacy paced-cycle deadline counters retain their
definitions. Neither a new metric nor a changed placement policy can clear a
failed legacy timing gate. Fold CPU placement uses only a measured pair of
allowed performance cores; kernel cpusets remain authoritative.

## Artifact identity and experiments

The immutable candidate archive records packaged, stripped provider-library
hashes separately from unstripped build artifacts. Installed APKs and every
experiment are checked against the immutable app hash. Separate accuracy parts
are combined only when their model, provider, device and implementation identities
match and every required comparison is present.

Every device experiment stops prior app/test processes, verifies camera release,
removes experiment forwarding and requires 30 continuous NORMAL seconds at skin
temperature at or below 32°C. The emulator uses NORMAL status because its sensor
is virtual. MODERATE pauses and critical aborts remain active. Cooling changes
are deferred. Inference timing and combined operation cannot advance ahead of
their accuracy, freshness, timing and watchdog prerequisites.
