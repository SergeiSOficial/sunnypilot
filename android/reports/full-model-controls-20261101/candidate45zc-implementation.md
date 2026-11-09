# Candidate45zc implementation

TX remains compiled out, USB is receive-only and physical engagement is disabled.
Original models, goldens and tolerances (`rtol=1e-3`, `atol=1e-4`) are unchanged.
The [candidate45za implementation](candidate45za-implementation.md) describes the
verified attention reduction repair and controls scheduling retained here.

## Exact convolution optimization

Candidate45zc changes regular FP32 convolution input loads to four-channel
vector loads and keeps the accumulator in registers. It retains the same
`ky/kx/ci` reduction order and correctly rounded `fma` at every step. It does
not reassociate the reduction, enable fast math or change FP16 boundaries.
Depthwise convolution and the verified half FDOT2 path retain their existing
implementations. `FOLD3_VECTOR_FMA_CONV` selects the new path; the strict scalar
implementation remains available with that macro absent.

The reproducible patch is [mnn-half-cast.patch](../../patches/mnn-half-cast.patch),
against MNN commit `7e3c6fb5225d3ccf22807ee641332669df34650d`. The immutable
candidate archive preserves its exact patch and provider libraries separately
from earlier candidates. All 795 host graph-conversion comparisons pass.

The [standalone operator probe](fold45za-conv-exact-probe.json), run while
candidate45za was installed, compares the actual new kernel source with its
strict scalar branch. All 36 synthetic cases are bit-for-bit equal. Cases
include three seeds, six geometries, both bias orders, odd channel tails,
padding, dilation, batch two and ReLU. GPU event medians for the three large
geometries change from 258.449/228.664/62.494 ms to
237.717/209.066/56.736 ms (8.02/8.57/9.21 percent improvement). These are
operator timings from a separately hashed diagnostic binary, not APK inference
latency, p99 measurements or full-model acceptance. Events are read after
submission without per-kernel waits. Compilation and transfers are separate.
Its [retry2 isolation record](fold45za-conv-exact-probe-retry2-isolation.json)
passes. Earlier interrupted attempts remain retained.

## Rejected faster arithmetic

Candidate45zb's sequential `mad` path passed the synthetic operator tolerances
but failed six of the 15 initial full-model comparisons. Road errors reached
0.125 and driver errors 0.00390625. It is rejected; recurrent/replay,
full-inference timing and combined operation did not advance. See the
[rejection record](candidate45zb-rejection.json). No result from that APK
qualifies candidate45zc or changes its tolerances.

## Build and host environment

The successful build and Java checks are in
[the retry build log](candidate45zc-build-retry1.log). The interrupted attempt
is preserved in `candidate45zc-build-interrupted.log`. The original host run
passed 174 of 176 checks; two native compilation checks failed because the
default Command Line Tools SDK advertised an architecture unsupported by its
linker. Both passed with the matched Xcode compiler and SDK, without a source
workaround. The complete suite is rerun with that same toolchain in
`candidate45zc-all-host.log`; the initial failures and focused retry are retained.

Repeated Mac maintenance sleeps interrupted earlier USB operations. Validation
uses a temporary, task-scoped `caffeinate -is` assertion, with no persistent
power-setting change. Device recovery requires fresh stop/camera/forwarding and
30-second thermal checks. Cooling changes remain deferred; all thermal pause
and abort conditions remain enforced.
