# Candidate45v: attention projection and controls release delay

TX stays compiled out, USB remains IN-only and physical engagement is disabled.
Cooling changes remain deferred. Existing thermal and experiment isolation gates remain active.

## Attention projection

Candidate45u passed all 159 driver comparisons exactly. Its road model passed
9/12 seed output arrays; 624 sequence/replay comparisons remained blocked.
Captured attention-value input `model/p_getitem_2` reproduces the road seed42
failure in the original host graph: 25 final output elements outside tolerance,
maximum absolute error 0.01171875. Captured attention logits or softmax do not
reproduce that final failure. Both attention MatMul kernels already match their
host paired-product arithmetic on captured inputs.

The value projection is an original FP16 NoTrans Gemm. Its pinned host preloads
bias and accumulates rounded FP32 pairs of exact half products; an ordinary
FP32 convolution/FMA reduction changes later FP16 rounding. The paired-product
calculation matches the projection exactly on original host inputs. The local
captured-input calculation passes projection tolerance; complete untapped
Android execution remains the acceptance gate.

The converter marks only static constant-weight/bias FP16 NoTrans Gemm with
unit alpha/beta and vector bias. Six road operations qualify; no driver
operation changes. The High/buffer GPU implementation reuses validated NC4HW4
staging and packed weights, preserves bias/pair/accumulator ordering, and leaves
the explicit final FP16 Cast in place. Unsupported shape, stride, group,
activation or precision is rejected. The reproducible MNN patch retains all
25 prior patched files. Original models, goldens and tolerances are unchanged.

Evidence: `road45u-value-causal.json`, `road45u-attention-matmul-causal.json`,
`road45u-softmax-causal.json`, `road45u-local-attention-matmul.json`,
`road45u-local-attention-value-matmul.json`, `road45u-local-dot2-attention.json`,
and `candidate45v-dot2-provenance.json`. These are diagnostic evidence only.

## Controls

Candidate45u's unchanged fixture averaged 100.002 Hz, work p99 3.451 ms,
complete-cycle p99 14.223 ms, and release-to-completion p99 6.483 ms. It had
zero work misses and two scheduled misses. The retained raw journal identifies
one 8.886 ms scheduler delay and one combined 4.503 ms scheduler delay plus
4.400 ms parameter/extension stage. Reporting cost was not the cause.

Candidate45v changes only the SM-F926B controls thread from Android DISPLAY
priority (-4) to URGENT_DISPLAY (-8). It retains ordinary unprivileged scheduling,
CPU placement checks, cancellation, stale inputs, watchdogs, raw cycle intervals
and every previous deadline definition. Other devices retain DISPLAY priority.
The actual applied priority is recorded by the service. Acceptance requires
the same fixture on this APK; a new priority cannot clear historical failures.
See `fold45u-controls-overruns.json` and `candidate45u-controls-summary.json`.

## Host checks

164 Android host checks pass; 1,719 Hyundai/safety/simulator checks pass and
202 are skipped. All 795 original-reference comparisons pass for derived host
graphs. Native MNN and Gradle results, device accuracy, installed hashes and
fixture results are recorded separately against the actual APK.
