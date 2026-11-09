# ncnn linear accumulation and controls timing

Candidate45zi (rejected controls experiment) APK SHA-256: `292136b0d550e2a3bc3910221d33165613d7f9668610f282026041b156bef7f3`. Final candidate45zj is built after reverting that experiment. APK SHA-256: `84c907b4957de484bcd4e72f0bd45436d8964085f46d35d3b3a005efa5d3eb58`.

## Implemented

- Targeted diagnostic `linear_100` ordered FP16 GEMV, preserving the original half-FMA accumulation, adjacent-pair reduction, and post-reduction bias boundaries. Explicit scalar input/weight/output packing prevents ncnn's pack4 or sum8 paths from bypassing the repair. Unsupported batched GEMM input is rejected. Other layers retain their existing implementation.
- Converter requires the exact expected operator, dimensions and unfused activation before marking the graph. Runtime rejects marked graphs without the corresponding native feature. Original model files and tolerances are unchanged; diagnostic graph/provider hashes invalidate previous qualification.
- **Rejected controls experiment:** candidate45zi bounded the Python GIL quantum to 1 ms. In the fresh paired emulator fixture, controls work p99 regressed from 1.250750 to 5.558833 ms, complete-cycle p99 from 11.409459 to 16.249542 ms, and scheduled deadline misses from 4 to 18. Source exchange latency also increased, so this run cannot isolate causality. The mitigation has been removed rather than claimed as an improvement.
- Final candidate45zj preserves the existing GIL policy and records its actual switch interval. Absolute pacing, input handling, heartbeat, stale-input checks, raw cycle timing and every deadline counter remain unchanged. Controls timing is still an unresolved gate.


## Validation

- **Passed:** final host suite, 188 tests; Gradle JVM tests and APK/instrumentation build; focused lint.
- **Passed:** converted graph external-I/O host reference, 159 comparisons (conversion correctness, not accelerator qualification).
- **Passed:** actual GLSL compiles in FP32, FP16-storage/FP32-arithmetic and FP16-storage/FP16-arithmetic configurations using pinned glslang.
- **Passed:** host execution of arithmetic extracted from the actual shader matches the original ONNX Gemm reference exactly for all 512 outputs on the captured ncnn input and three additional seeded vectors. The previous captured GPU output failed 279/512. This isolates the repaired arithmetic. Separately, actual software Vulkan execution of the marked layer matches all 512 outputs exactly, with packing enabled; a CPU-only attempt is rejected. The probe provider hash matches the final candidate. Software Vulkan establishes shader correctness, not Fold 3 speed or qualification.
- **Blocked:** Fold 3 GPU execution, full 795-comparison qualification and device performance: phone disconnected. Models remain diagnostic and unqualified.
- **Passed:** final installed emulator APK hash; 28 instrumentation tests; 9 of 10 experiment groups including process-chain fixtures, lifecycle, ADB recovery, IPC, MetaDrive cancellation/stale inputs and thermal safeguards. Every isolation check and final cleanup passed; emulator exited with no app processes, camera ownership or experiment forwards.
- **Failed:** native emulator camera timing still reports future capture timestamps. No arrival-time substitution was introduced.
- **Failed:** final emulator controls timing: work p99 8.648375 ms, complete-cycle p99 19.644417 ms, release-to-completion p99 15.026346 ms, mean 96.4064 Hz, 57 scheduled deadline misses and 96 skipped slots. Reporting p99 is 0.335334 ms; input handling, publishing and scheduler delay remain significant. The fixture functionally passed but timing qualification did not. Paired candidate45zg/45zi controls reports retain their identical fixture hash and differing source latency; they do not establish a final speed improvement.
- **Failed:** untapped complete driver model on software Vulkan, seed 42: 14/553 mismatched outputs, max absolute error 0.0046749115, all finite. Remaining seeds and performance stages were not advanced.
- **Passed local repair / failed end-to-end:** captured full-model `linear_100` input through the repaired shader agrees exactly on all 512 outputs with the original host Gemm. The first mapped end-to-end failure is still `linear_100` (3/512, max 0.0009765625), because its input already differs from the host; later heads add divergence. The preceding mapped Cast remains within tolerance but differs by up to 0.00048828125. This evidence establishes the repaired local accumulation, not complete graph accuracy. Trace results are diagnostic, separately hashed, and cannot replace untapped acceptance.
- **Passed:** full-model and trace-run cleanup, no app/probe processes, camera ownership or forwards, NORMAL cooldown, emulator shutdown. Physical promotion still requires all original-reference comparisons on the Fold 3.

Vehicle TX stays compiled out and engagement disabled. No cooling optimization. Phone remains unmounted, Panda disconnected and lens-cover proof deferred.

`candidate45zh` is excluded: its archive was created before the build finished and paired the old APK with new diagnostic artifacts. Zero experiments completed against it. Its exclusion and cleanup records are preserved; candidate45zi was archived only after successful build and package verification.

Reproduction: `reproduce.sh`, `check_linear_shader.py`, `check_linear_arithmetic.py`, and `run_emulator_45zj.py`. The runner verifies the installed APK hash and wraps each experiment in process/camera/forward cleanup plus the existing cooldown gate. Earlier failures and candidate archives remain intact.

The isolated software-Vulkan probe uses the candidate45zi APK for artifact identity while executing a separately hashed standalone binary with the same patched provider. It never starts the driving runtime and cannot qualify a physical backend. Its cleanup and result are separate from final candidate45zj instrumentation.

Additional diagnostic reproduction, after functional-run cleanup: `run_full_ncnn_software.py`, then `run_head_ncnn_software.py`, `android/scripts/compare_ncnn_trace.py ncnn-software-head-seed42.json` (use the report absolute path), and `diagnose_ncnn_head.py`. Preserve distinct result directories when repeating experiments.
