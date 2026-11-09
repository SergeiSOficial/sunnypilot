# Reproduce the host arithmetic checks

## Candidate32 exact half-dot optimization

`fold3_host_math.py` replaces FMA with separate FP32 multiplication and addition
only where both operands are binary16 values. Their finite products fit exactly
in FP32, including products of half subnormals. The two-product pair sum and
the subsequent accumulator addition remain separate rounding boundaries;
the odd-length tail keeps its original position. FP32 convolution, FP32
MatMul and the normalization repair retain their existing arithmetic.

The independent regression covers all 65,536 half encodings, replaces
non-finite encodings with signed maximum finite values, and checks MatMul and
Gemm against a wider-precision reference with exact integer-bit comparisons.
The [complete host model gate](candidate32-host-reference-models.json) passes
the unchanged original seeds, recurrence and camera replay.

The [native OpenCL half-input probe](fold-half-product-probe-candidate31.json)
also finds zero differing outputs across 65,536 outputs with 128 terms each.
Its pair-FMA and pair-add means are **21.597/0.636 ms**. This bounded probe ran
with candidate31 installed; its independently compiled ELF and source hashes
are recorded [separately](opencl-half-product-probe-artifact.json).
The earlier [FP32-input probe](fold-arithmetic-probe-v2-candidate27.json) found
50,920 different outputs when FMA was replaced. That broader replacement is
not part of candidate32.

The [candidate32 serialized kernel profile](candidate32-serialized-kernel-profile.json)
records road `host_half_dot2` event time of **4.223 ms**, versus **80.190 ms**
in [candidate29](candidate29-serialized-kernel-profile.json). FP32 convolution
still accounts for **658.856 ms road / 362.301 ms driver** in those checkpoints.
Serialization, compilation warmup and tracing perturb these figures; they are
diagnostic costs, not end-to-end inference or sustained performance results.
The [reproduction commands](candidate32-reproduce.md) run the separate full
device numerical and timing gates without serialized checkpoints.

The [complete candidate32 Fold 3 gate](candidate32-model-accuracy-gate.json)
passes all **795 output comparisons exactly**: both models, seeds 42/7/123,
96 recurrent frames and 60 replay frames per model. Road p50/p99 is
**850.809/1,021.535 ms**, versus candidate31 **941.232/1,133.891 ms**.
Driver p50/p99 is **496.048/559.414 ms**, versus **497.369/560.536 ms**.
These are observations from separately cooled runs, not a controlled A/B
benchmark. Both models still miss all 100 timed 50 ms deadlines, so accurate
OpenCL inference remains diagnostic and cannot be selected for live use.

## Candidate31 normalization square-root repair

The [candidate30 recurrent trace](candidate30-road-recurrent-trace-comparison.json)
locates the first seed-7/frame-10 difference in `LayerNormalization` at node
442. Inputs, feedback states and the fresh runner's reproduction of the JIT
outputs match exactly. The first tolerance failure occurs downstream at node
471; the final predictions have 14 mismatched elements. The host traced graph
still matches the unchanged original goldens.

`rounded_sqrt` corrects the normalization standard deviation with integer
midpoint comparisons. Positive normal FP32 inputs use the native square root
as a starting point; exceptional inputs retain the platform result. The
[OpenCL accuracy specification](https://registry.khronos.org/OpenCL/specs/3.0-unified/pdf/OpenCL_C.pdf)
allows a three-ulp error for native `sqrt`. Four correction steps cover that
bound without deciding a rounding step with approximate division or floating
multiplication. The standalone device diagnostic in
[candidate28](fold-sqrt-probe-candidate28.json) previously found 3,522 native
rounding differences and zero differences after correction across 65,536
positive normal FP32 inputs. The UOp implementation also passes an independent
65,536-element host regression, including IEEE fallback cases.

The [candidate31 recurrent comparison](candidate31-road-recurrent-trace-comparison.json)
now matches all 531 tensors and final outputs exactly for the same failing
frame. This is a diagnostic trace: retained tensor reads can affect scheduling,
and it does not replace the complete device model gate. The
[complete host regression](candidate31-host-reference-models.json) passes all
three seeds, 96 recurrent frames and 60 camera-replay frames per model at the
unchanged original hashes and `rtol=1e-3, atol=1e-4`.

The subsequent [complete Fold 3 accuracy gate](candidate31-model-accuracy-gate.json)
also passes both original models: all three seeds, 96 recurrent frames and 60
camera-replay frames per model. Every compared output is exact. This resolves
the observed numerical blocker, but the same runs fail timing: road p99
**1,133.89 ms**, driver p99 **560.54 ms**, with 100/100 misses per model against
50 ms. The single-model profiles remain diagnostic and cannot select a live
backend. Vehicle transmission and physical engagement remain disabled.

```sh
.venv/bin/python -m unittest android.tests.test_host_math
.venv/bin/python android/scripts/validate_host_math.py \
  --report android/reports/blocker-fixes/candidate31-host-reference-models.json
.venv/bin/python android/scripts/compare_camera_trace.py \
  --model driving_supercombo --recurrent \
  --archive android/build/candidates/candidate31/recurrent-trace/model-camera-trace-driving_supercombo.npz \
  --report android/reports/blocker-fixes/candidate31-road-recurrent-trace-comparison.json
```

## Candidate29 bit-preserving concatenation

The [candidate28 trace](candidate28-road-concat-diagnostic.json) located the
first road-model difference in `Concat`: two nonzero half subnormal inputs
became zero. The pinned `Tensor.cat` adds zero-padded inputs of unequal sizes;
Adreno's half arithmetic flushes those values. `fold3_host_math.py` now
concatenates the corresponding integer bits and bitcasts the result back.
The operation also preserves signed zero and NaN payloads, with focused
float16/float32 regression coverage. It does not change model precision.

The [candidate29 comparison](candidate29-concat-repair.json) now matches all
531 inspected road tensors exactly on replay frame 0, versus 527 before the
repair. The complete host gate also passes both original models: all three
seeds, all 96 recurrent frames per model and all 60 replay frames per model.
Retained tensor reads can change scheduling; the trace is diagnostic and
does not replace the separately recorded device sequences or timing gates.

```sh
.venv/bin/python -m unittest android.tests.test_host_math android.tests.test_model_sequences
.venv/bin/python android/scripts/validate_host_math.py \
  --report android/reports/blocker-fixes/candidate29-host-reference-models.json
.venv/bin/python android/scripts/compare_camera_trace.py \
  --model driving_supercombo \
  --archive android/build/candidates/candidate29/camera-trace/model-camera-trace-driving_supercombo.npz \
  --report android/reports/blocker-fixes/candidate29-road-camera-trace-comparison.json
```

The new `host-fusion-cl-road` and `host-fusion-cl-driver` diagnostic profiles
allow each original model's complete tests to start after its own cooldown.
Neither profile can write a live-backend qualification record. Original model
hashes, goldens and `rtol=1e-3, atol=1e-4` remain unchanged.

The [complete candidate29 device runs](candidate29-full-model-summary.json)
pass both models' three individual seeds and both 60-frame camera replays.
Driver monitoring also passes all 96 recurrent frames. Road-model recurrence
still fails frames 25/27 (seed 42), 10 (seed 7), and 23/24 (seed 123), versus
23/24/18 failed frames respectively in candidate27. All recurrent feedback
tensors match exactly; the remaining differences affect final predictions.
The original device timing gate fails: road/driver p99 is 1121.55/605.06 ms,
with 100 of 100 misses per model. These are cold, separate diagnostic runs;
they are not a controlled speed comparison with earlier inaccurate backends.

## Candidate27 device division repair

The [host qualification](candidate27-host-reference-models.json) passes both
original models: seeds 42/7/123, 32 recurrent frames per seed and the pinned
60-frame camera replay. No model, golden or tolerance was changed.

Candidate25's device replay trace located the first numerical difference in
the division inside MLAS Tanh at node 153. One quotient rounded down instead
of up, eventually producing six failing final driver-monitoring elements.
Candidate26's second residual correction did not repair the midpoint; its
failed trace is retained. Candidate27 classifies the exact FMA residual against
half the adjacent-float gap and applies ties-to-even explicitly. The
[same replay-frame comparison](candidate27-camera-trace-comparison.json) now
has 435 of 437 tensors exactly equal, all 437 within the original tolerance,
and all 553 final half outputs exactly equal to the original reference.

The [Fold 3 full model run](fold-models-candidate27.json) passes driver
monitoring's three seeds, all 96 recurrent frames and all 60 replay frames.
It still fails timing: p99 is 506.85 ms, with 100 of 100 deadlines missed.
The road model passes the three single-frame seeds but fails recurrent and
camera replay outputs. MODERATE thermal protection stops the run before road
timing completes. This remains an unqualified diagnostic backend.

```sh
.venv/bin/python android/scripts/validate_host_math.py \
  --report android/reports/blocker-fixes/candidate27-host-reference-models.json
.venv/bin/python android/scripts/compare_camera_trace.py \
  --archive android/build/candidates/candidate27/camera-trace/model-camera-trace-dmonitoring_model.npz \
  --report android/reports/blocker-fixes/candidate27-camera-trace-comparison.json
```

## Candidate24 production arithmetic and scheduling

The production `fold3_host_math.py` kernels now pass both original models on
the host GPU: seeds 42/7/123, 32 recurrent frames per seed, and every frame of
the pinned 60-frame camera replay. That is 312 sequence/replay comparisons,
in addition to the six seed cases. The [complete result](candidate24-host-reference-models.json)
records the source, graph and unchanged reference hashes. This is a host
numerical regression result; Android OpenCL and its timing require their own
qualification.

```sh
.venv/bin/python android/scripts/validate_host_math.py \
  --report android/reports/blocker-fixes/candidate24-host-reference-models.json
.venv/bin/python -m unittest android.tests.test_host_math android.tests.test_opencl_precision
```

Intermediate tensor traces identified several distinct rounding boundaries:
convolution bias/residual placement, half GEMV packet reduction, paired half
attention dot products, Welford normalization, aligned Eigen reductions, MLAS
activation/Exp polynomials and shape-dependent scalar Pow. The kernels preserve
these boundaries rather than converting the whole model to FP32. Fourteen
focused arithmetic checks compare independent ONNX Runtime CPU micrographs
with the host GPU, including broadcast/layout cases and non-finite inputs.
Original models, goldens and `rtol=1e-3, atol=1e-4` remain unchanged.

Candidate23's serialized phone OpenCL trace also passes both models for seed 42
at those tolerances. It exposes a scheduling problem: the ordered custom
kernels launch one scalar per workgroup. Candidate24 groups independent scalar
outputs into up to 32 local lanes while preserving each output's accumulation
order. All fourteen arithmetic checks and the complete host gate pass after
this change. This evidence does not replace the phone's three-seed, recurrent,
replay or performance gates. Serialized event times are diagnostic and do not
establish sustained inference latency.

The rejected automatic-vectorization and fixed-axis scheduling experiments are
retained as `candidate24-*-rejected.*`. They were not packaged. Opaque scalar
arithmetic requires explicit output lanes; the generic optimizer could emit
invalid vector casts or choose an axis whose factor could not be split.

The earlier passing [candidate22 host gate](candidate22-host-reference-models.json)
and failing numerical probes remain alongside the current result:
`candidate22-host-models.json`, `candidate22-host-eigen-exp-models.json`,
`candidate22-host-aligned-models.json` and
`candidate22-host-complete-models.json`.

## Historical candidate21 prototypes

These experiments locate rounding differences against the pinned original
reference. They run on the Mac's Metal backend or NumPy; they are **not Android
OpenCL qualification**. Only seed 42 is used for the complete prototype traces.
The original model files, three-seed goldens, recurrent/replay fixtures and
`rtol=1e-3, atol=1e-4` remain unchanged. TX is disabled.

The [source archive](host-numerical-diagnostics.zip) preserves the actual
experimental kernels and probes, with portable repository-root discovery.
[Source hashes and model identities](host-numerical-diagnostics-source.json)
identify its contents. These scripts write diagnostic reports; they do not
install an APK or select a live inference backend.

From the repository root with the pinned host environment and prepared model
references described in [the Android build instructions](../../README.md):

```sh
unzip -o android/reports/blocker-fixes/host-numerical-diagnostics.zip -d android/build
.venv/bin/python android/scripts/model_trace.py --optimized --limit 1000 \
  --output-directory android/build/trace-complete
.venv/bin/python android/build/diagnose-fused-fp32.py
.venv/bin/python android/build/prototype-host-conv.py
.venv/bin/python android/build/prototype-host-conv.py --full --complete
.venv/bin/python android/build/diagnose-logistic.py
.venv/bin/python android/build/prototype-mlas-functions.py
```

The isolated exporter validates the traced finals against the unchanged original
goldens. It exports all 437/531 observable DM/driving tensors without updating
packaged traces or host-fusion models. Passing `--output-directory` inside the
packaged reference directory is rejected. The actual isolated-export check and
unchanged packaged-reference hashes are recorded in
[candidate21-isolated-trace-export.json](candidate21-isolated-trace-export.json).

The first probe tries accumulation order, bias placement and FMA policies for
256 sampled outputs of each first NHWC convolution. The ordered convolution
prototype then evaluates whole tensors using one float32 FMA accumulator per
output. The MLAS prototype additionally tries the pinned activation polynomial,
Metal precise division and bias-last ordinary NCHW convolution. It deliberately
keeps the original half-rounding and Gemm implementation from `fold3_opencl`.
Coefficients follow the locally pinned ONNX Runtime 1.31.1 MLAS `tanh.cpp` and
`logistic.cpp`; that runtime's MIT license is retained in the native source tree.

These historical results were diagnostic: four observed NHWC tensors matched exactly; the full
ordered/MLAS single-seed DM final meets tolerance but intermediate failures
remained, and driving still had 213 failing final elements. No production kernel
was changed in candidate21 from those host-only results. Android must independently pass all
three seeds, recurrent sequences, the pinned replay and performance gates before
any backend can be accepted. The candidate21 APK remains the previously tested
artifact; these host scripts are not packaged into it.
