# Candidate45s numerical recovery

TX remains compiled out, USB is IN-only and physical engagement is disabled. Cooling optimization is deferred; existing thermal safeguards and experiment cleanup remain mandatory.

## Root causes and minimal repairs

- Candidate45r retained REALDIV in the converted graph, but MNN GeometryBinary lowered the road per-channel broadcast to a Loop. The Loop OpenCL expression bypassed the strict division helper. The pinned patch now uses the same verified scalar correction in High/buffer loop execution, including fused loop outputs. Other precision modes are unchanged.
- The physical Adreno arithmetic probe tested 1,024 values under both default and actual MNN compiler flags. Plain division differed by one FP32 ULP; the existing scalar and vector corrections were exact. Source, executable, inputs, results and isolation records are retained under `diagnostics/` and `fold45r-division-probe-static-isolation.json`. This probe is not model qualification.
- Driver seed 123 frame 27 traced the final half-output error to upstream FP32 Gemm accumulation in nodes 426 and 427. Given actual captured inputs, ordered ascending-K FMA followed by bias and ReLU matched both host outputs exactly; the generic GPU kernels differed by up to 2.38e-7. This small difference can cross the following FP16 midpoint.
- Static original FP32 GEMV operations are tagged through conversion to reuse the existing verified 1x1 ordered-convolution kernel. Tags require original provenance, batch one, constant FP32 weights, transA=0, transB=1, alpha=beta=1 and matching static dimensions. Unsupported shapes/precision remain unchanged. Nodes, weights, original casts and external interfaces remain unchanged.

## Acceptance

Original models, goldens, rtol=1e-3, atol=1e-4 and all 795 comparisons remain mandatory. Diagnostic tensor taps and local arithmetic probes cannot qualify full untapped graphs. Neither model is promoted until all applicable initial, recurrent and replay comparisons and no-CPU-fallback checks pass. Performance stages remain gated by correctness, fresh inputs and watchdog/timing results.

Candidate35, candidate40, candidate44 and all intermediate candidate45 artifacts and failures remain preserved. The phone is unmounted, the Panda disconnected and independent rear-lens cover trials deferred.

## Actual candidate45s result

The installed application SHA-256 was `e23bb4fa65425f3976f82a704db3ad4cf1d2e5baa0a405b10f42fb2554b54903`, verified on both devices. On the Fold 3, all 159 driver comparisons passed with no CPU fallback. Android reached MODERATE thermal status during the following timing stage; the experiment aborted before road validation. Road's 636 comparisons and accelerator timing therefore remain blocked for this artifact. Cleanup and camera release passed. See `fold45s-mnn-buffer.json` and `fold45s-mnn-buffer-isolation.json`; these are retained failures, not a qualified backend.
