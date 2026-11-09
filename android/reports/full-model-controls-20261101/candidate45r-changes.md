# Candidate45r implementation and evidence

Vehicle TX is compiled out, USB remains IN-only and physical engagement remains disabled. Cooling optimization is deferred; thermal pauses and aborts remain active.

## Numerical correction

The candidate45p OpenCL trace first differed at road normalization division by 63.75: 386,936 of 786,432 values differed by at most one FP32 ULP. The preceding subtraction was exact. The other 14 locally checked operations in that trace window were exact given their actual GPU inputs. Inspection of the converted MNN operator showed that its constant-division optimizer had replaced the division with multiplication by a rounded reciprocal. That bypassed the native strict division kernel.

The reproducible pinned MNN patch now retains constant division as REALDIV. A regression invokes the actual MNN converter on four constant divisors and checks the serialized operation. It failed against the previous converter and passed after explicitly rebuilding MNN. Original models, goldens, tolerances and recurrent interfaces remain unchanged. Host graph conversion still passes all 795 original-reference comparisons. Host conversion does not establish accelerated execution accuracy.

The new Gradle identity check rejects stale native builds and converted graphs. The first provisional build was stopped before installation after this issue was detected; it is retained as an interrupted build log. Candidate45q was archived but never installed.

## Controls reporting

Controls phase changes between running and waiting previously crossed JNI on each transition. The bounded reporting path now emits routine heartbeats at 4 Hz and emits failures immediately. Other daemons retain immediate phase-change reporting. Absolute 10 ms slot scheduling, nonblocking latest inputs, missed-slot accounting, raw intervals, legacy complete-cycle metrics and watchdog reporting are preserved.

The selected-frame MNN trace tooling now binds raw inputs to their hashes and can reproduce a specified driver recurrent case. Diagnostic tracing is not used as full-model acceptance or as an accelerator performance result.

## Host verification

- 152 focused Android/runtime/converter/reporting tests passed.
- 1,921 Hyundai safety, vehicle and simulator checks completed successfully: 1,719 passed and 202 skipped.
- All 795 original-reference converted-graph comparisons passed.
- The initial host run retained one obsolete test expectation for immediate controls phase changes; the corrected test explicitly covers immediate phase changes on a non-controls daemon.

## Device gates

Installed application and instrumentation hashes matched candidate45r on both devices. Untapped physical MNN execution passed 167 of 171 comparisons run, failed four and blocked 624. Driver monitoring passed 158/159: seed 123 frame 27 differed at one output. Road outputs failed all three initial seeds; there was no CPU fallback or unexpected non-finite output. No model was promoted.

The matched physical controls fixture remained failed: work p99 8.706 ms, complete-cycle p99 17.000 ms, 15 work deadline misses, 1,369 legacy cycle misses and 69 scheduled release misses. The intended CPU pair was unavailable (core 5 offline); startup affinity was [0,1,2,4,6]. This differs from the earlier 45o run and is retained, not treated as an improvement. Emulator validation passed nine groups; camera timing failed because invalid emulator capture timestamps were rejected.

A separately hashed native OpenCL probe proved both strict division helpers exact for 1,024 representative values under the actual MNN compile flags. The model still bypassed the helper: GeometryBinary lowered the per-channel broadcast to Loop, whose division remained uncorrected. Driver tracing isolated FP32 Gemm nodes 426/427 before a half cast; ordered FMA accumulation with bias last matched the host exactly on captured GPU inputs. These findings drive candidate45s and do not qualify candidate45r. Earlier candidate failures remain retained. The phone is unmounted, the Panda is disconnected and independent rear lens-cover trials are deferred.
