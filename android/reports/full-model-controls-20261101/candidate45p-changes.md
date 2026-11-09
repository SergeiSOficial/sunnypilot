# Candidate45p numerical and scheduling changes

This continues the full-model recovery on `codex/fold3-android-port`. Candidate35,
candidate40, candidate44 and every candidate45 subexperiment remain separate
artifacts. Original models, goldens, `rtol=1e-3` and `atol=1e-4` are unchanged.

The actual MNN buffer trace identifies the driver's first propagated difference
at `InsertedPrecisionFreeCast_add_1`. The preceding convolution and residual
inputs match the host. MNN's Add also matches their elementwise sum exactly.
The difference comes from decomposing the pinned host's NCHW FusedConv: MLAS
adds the residual before its bias pass, whereas the previous portable graph
included bias in Conv before adding the residual. On captured inputs, restoring
residual-before-bias removes all 4,318 differences at this tensor. See
`driver45m-residual-bias-order-probe.json` and
`driver45m-first-residual-probe.json`. This is diagnosis, not full-model acceptance.

`reference_precision.portable` now preserves that order for the proven FP32
NCHW fused path. Bias is broadcast over channels explicitly. NHWC bias-first
fusion and FP16 boundaries keep their existing behavior. A numerical regression
compares the unfused converted graph against the original FusedConv, including
activation and a cancellation case where the old order loses the bias.

Road normalization also exposed an Adreno division-rounding difference. The
High-precision MNN buffer kernel now uses the reference implementation's FMA
residual correction and midpoint tie handling for FP32 division. Other precision
modes retain upstream behavior. `road45m-division-residual-probe.json` records
the captured-input diagnostic; candidate45n separately tests this native fix
before the residual conversion change. Non-finite operands retain IEEE handling.

Earlier candidate45 repairs remain: explicit GPU FP16 rounding boundaries,
host GEMV/Tanh/Conv accumulation, depthwise layout and provenance, scalar powers,
spatial-mean order, session resize failure rejection, absolute 10 ms Android
controls slots, expired-slot skipping and bounded reporting to a separate
process. The measured Fold 3 controls-core pair is applied only to the calling
thread when the device topology and allowed mask support it. The policy and
any failure are reported; it changes no control calculations or watchdog limits.

Host graph checks and local tensor probes do not qualify perception. Untapped
on-device seeds, recurrent sequences and the pinned replay remain the numerical
acceptance gate. Traced readbacks are excluded from timing. APK/model/provider
hashes and device identity bind qualification records.

TX is compiled out, USB is IN-only and physical engagement is disabled. Cooling
optimization is deferred. Thermal pauses, aborts, camera freshness/synchronization
limits and experiment cleanup/cooldown remain enforced.

## Device result and converter coverage failure

The untapped Fold 3 model comparison failed. Candidate45o has exactly the same MNN I/O graphs as candidate45n: optimized host graphs omit internal value_info, so the FP32-only residual rewrite never applied. The original regression had an externally typed input and missed this case. Candidate45p uses the same-type Conv weight initializer when the internal input type is absent, and adds an untyped internal-input regression. Candidate45o is retained as a failed artifact; its host 795 comparisons do not establish that the intended rewrite occurred.

Candidate45p converter coverage is checked against the actual saved MNN I/O graphs: driver 10 and road 12 residual/bias additions. The added regression includes an internal input without value_info. See `candidate45p-converter-coverage.json`. Native provider libraries are unchanged from candidate45n.
