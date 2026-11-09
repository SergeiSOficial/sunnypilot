# Candidate45zd controls CPU selection

Vehicle transmission remains compiled out and engagement disabled. Models,
goldens, tolerances and candidate45zc's accurate MNN/OpenCL libraries are
unchanged. The [candidate45zc implementation](candidate45zc-implementation.md)
records the inherited arithmetic repairs and exact convolution optimization.

The [candidate45zc steady fixture](fold-candidate45zc-chain-device-steady.json)
reports an allowed controls CPU mask `[0, 1, 2, 5, 6]`. Policy v2 accepts only
`[4, 6]` and `[4, 5]`, so it leaves the loop on its original mask when Android
removes core4. Work p99 is 6.545 ms and complete-cycle p99 is 15.493 ms, with
1,364 complete-cycle and 12 scheduled-completion misses. That failure remains
retained under its actual APK.

Policy v3 keeps those preferences and adds `[5, 6]` as the final choice. Both
cores must be in the calling thread's allowed mask, have equal positive
capacity and exceed at least one available lower-capacity core. It does not
expand cpusets, select the prime core, change other workers or override Android
thermal/hotplug decisions. The actual resulting affinity is verified; missing
topology or permission failures remain visible. Other devices and explicit
debug affinity fixtures retain their behavior.

A focused regression check uses the exact observed core4-unavailable mask and
also rejects a mismatched-capacity pair. All 23 scheduling, affinity and
reporting checks pass, followed by the complete 177-test host suite. The complete 795-comparison suite has now passed on candidate45zd's actual APK,
with original tolerances and no CPU fallback. All six Fold functional groups
pass; accelerator and complete-cycle controls performance remain unqualified.

The report generator now treats missing model evidence and pending device
groups as blocked, requires nonempty evidence before declaring no fallback,
and requires complete original-reference accuracy before accelerator timing
can qualify. Focused report-generator checks preserve historical results and
reject mixed-APK evidence. These reporting changes do not alter the APK.
