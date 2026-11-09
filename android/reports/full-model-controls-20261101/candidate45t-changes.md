# Candidate45t staged validation

TX remains compiled out, USB is IN-only and physical engagement is disabled. Cooling optimization remains deferred; existing thermal safeguards and cleanup remain enforced.

Candidate45s repaired Loop-buffer division and original FP32 GEMV accumulation. Its 159 driver comparisons passed; MODERATE thermal status interrupted timing before road validation. That failure remains in the candidate45s reports.

Candidate45t retains those native libraries and converted graphs. Separate road and driver accuracy profiles run the unchanged three initial seeds, three 32-frame recurrent sequences and the pinned 60-frame replay. Accuracy records bind the APK, original and converted models, reference assets, implementation, provider libraries, driver and device fingerprint. Traced sessions and CPU fallback cannot supply those records.

Timing is a separate stage and cannot create a model session until both current accuracy records pass all 795 comparisons at rtol=1e-3 and atol=1e-4. Numerical evidence alone cannot qualify live inference. The default full run now completes both models' numerical stages before either timing stage.

Six focused tests cover staged ordering, artifact invalidation, incomplete evidence, no fallback, diagnostic traces and the separation between numerical and performance acceptance. The complete host suite passed 159 tests. Hyundai and simulator checks ran 1,921 tests, with 1,719 passed and 202 skipped. Device results will be recorded separately against this candidate's actual APK hash.
