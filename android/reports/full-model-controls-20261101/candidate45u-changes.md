# Candidate45u: scoped road sigmoid repair

Vehicle TX is compiled out; physical engagement remains disabled. Cooling
optimization is deferred. Existing thermal pauses and aborts remain enforced.

## Numerical cause and repair

Candidate45t's untapped road model failed all three seed output comparisons.
Actual MNN OpenCL buffer traces match the host exactly through the convolution
feeding `InsertedPrecisionFreeCast_model/sigmoid`. The current sigmoid differs
at 807/1024 values (maximum absolute error 1.1920928955078125e-7). Its own error
is within tolerance, but downstream FP16 boundaries amplify it. Injecting that
captured sigmoid into the host reproduces the spatial mean and failing
`model/linear_35` values exactly, including the tolerance failure.

`prepare_accel_models.py` now tags only a sigmoid with original FP32 provenance,
static dimensions and complete float4s. The road tag derives from the pinned
host's FP32 `FusedConv` sigmoid activation, not a guessed tensor layout. External
inputs/outputs, original FP16 boundaries, models and goldens are preserved.
MNN's High/buffer path uses the pinned ARM64 MLAS logistic polynomial, explicit
FMAs and the already verified correctly rounded division for this tagged
operation. Other sigmoid operations and the image backend retain their behavior.

Evidence:

- `road45t-trace288-all.json`: first small difference and downstream failure.
- `road45t-sigmoid-causal.json`: exact causal replay with actual GPU inputs.
- `diagnostics/sigmoid-host-polynomial.json`: all 1024 polynomial values exact.
- `diagnostics/fold45t-sigmoid-probe.json`: the production helper matches all
  1024 values on the Fold 3 GPU, with default and MNN compiler flags. Both
  ordinary exponential implementations retain their measured rounding error.
- `fold45t-sigmoid-probe-isolation.json`: cleanup passed; actual installed APK
  identity retained. This standalone probe does not qualify a model or backend.

The patch is reproducible against MNN 3.6.2 revision
`7e3c6fb5225d3ccf22807ee641332669df34650d`, using
`android/patches/mnn-half-cast.patch`. Converter tests check original precision,
vector size, provenance, external contracts, collisions and the actual MNN
UnaryOp/SIGMOID tag.

## Controls placement

The existing measured performance-core policy rejected the whole placement
when CPU5 was offline. Policy v2 prefers the separately measured CPU4/CPU6
pair and retains CPU4/CPU5 as a fallback. It checks the allowed CPU mask and
capacity topology, changes only the calling controls thread, and records a
failure when neither measured pair is available. Android scheduling, thermal
management and all deadline counters remain authoritative.

On candidate45t, the unchanged 30-second fixture with diagnostic CPU4/CPU6
placement measured work p99 of 3.17 ms versus 5.98 ms without placement, and
release-to-completion p99 of 6.44 ms versus 10.83 ms. Complete-cycle p99
improved from 17.05 ms to 14.01 ms but still failed. These are candidate45t
diagnostic results, not candidate45u acceptance evidence. See
`fold45t-affinity46.json` and its isolation record.

## Host validation

- Android host suite: 162 passed (`candidate45u-host.log`).
- Hyundai safety, vehicle behavior and simulator protocol: 1,719 passed,
  202 skipped (`candidate45u-safety.log`).
- Controls placement checks: 8 passed
  (`candidate45u-controls-affinity-tests.log`).
- Focused sigmoid/converter checks: 2 passed
  (`candidate45u-sigmoid-converter-tests.log`).
- Original and derived graph comparisons: 795 passed, including all three
  seeds, recurrent sequences and the pinned 60-frame replay
  (`candidate45u-conversion.log`, candidate graph manifest).

These host and primitive checks do not establish complete on-device model
accuracy or accelerator performance. Device results and installed hashes are
recorded separately for the actual APK.

## Current untapped device accuracy

The installed candidate45u APK is
`e34f7d2979bd37b1457b8f83052bdf9436c4b10c491097988078822c87b29319`.
Its driver model passed all 159 original-reference comparisons exactly,
including three initial seeds, three recurrent sequences and the pinned
60-frame replay, with every audited operation assigned to OpenCL. This is
accuracy evidence only; accelerator timing remains blocked until both models
pass. See `fold45u-mnn-driver-accuracy.json` and its isolation record.

The road model passed 9 of its 12 initial output comparisons. All feature,
image and desire recurrent-state outputs are exact, but the three final
output arrays still fail. Their mismatch counts fell from candidate45t's
58/123/225 to 25/66/44 for seeds 42/7/123. The remaining 624 road recurrent
and replay comparisons were blocked, as required. The post-repair trace
through linear5/linear6, vector normalization and linear8/linear10 has exact
rounded outputs; the remaining head calculations require further diagnosis.
See `fold45u-mnn-road-accuracy.json` and `road45u-trace416-all.json`.
