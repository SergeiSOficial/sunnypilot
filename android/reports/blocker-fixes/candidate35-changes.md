# Candidate35 changes and qualification boundaries

Candidate35 APK SHA-256:
`38645e91f806076517de1497cdb6b5c0cbf4873cfecf5efabb980297a014ad51`.
The [artifact record](candidate35-artifact.json) identifies the uncommitted source,
unchanged native dependencies, app and instrumentation APKs. See the final
[validation report](candidate35-validation.md) for actual device results.

- Camera callbacks, GL acquisition and publication have separate bounded workers.
  Three owned NV12 buffers prevent a slow consumer from observing overwritten
  data; the newest pending frame replaces an older pending frame. Counters retain
  superseded publications, duplicate callbacks and readback drops. A 120 ms slow
  consumer test checks acquisition progress, ownership and teardown.
- Camera outputs explicitly request the sensor timestamp base and disable readout
  timestamps on API 34+. Android's default SurfaceTexture delivery re-spaces frames
  for smoother preview; explicitly selecting SENSOR avoids that added delivery
  delay. Raw sensor, elapsed-realtime and monotonic clocks remain recorded;
  invalid capture times are rejected, never replaced by arrival times.
  [Android OutputConfiguration documentation](https://developer.android.com/reference/android/hardware/camera2/params/OutputConfiguration#TIMESTAMP_BASE_DEFAULT).
- Diagnostic capture profiles compare recording, preview, 720p preview and a
  two-buffer reader. Continuous-readback mode exercises conversion, fences, PBOs,
  owned buffers and publication on every preview frame. It does not certify the
  physical origin of Samsung's multiview thumbnail. Live capture keeps the
  recording profile and the existing 100 ms / 20 ms freshness and skew limits.
- Runtime reporting snapshots a bounded queue once per second. A background
  writer sorts percentiles and atomically writes JSON. Heartbeats remain active,
  reporting failures propagate, and all work and complete-cycle misses are
  counted. Tests cover blocked disk I/O, failed writes and flush completion.
- Debug-only, authenticated, bounded on-device replay publishes the existing
  source fixtures on 100 Hz deadlines. The supervised vehicle, localization,
  calibration, learning, radar, planning and selfdrive chain must generate the
  derived outputs. This removes per-packet ADB pacing from timing measurements.
  Factory-ACC cancellation, stale CAN and bridge disconnect remain fault cases.
- Strict reference arithmetic keeps the original ordered FMA and rounding rules.
  Candidate33's four-channel strict convolution passed all 795 comparisons but
  regressed to road/driver p99 2,109.00 / 1,203.29 ms; it was reverted. Separate
  native-MAD profiles remain diagnostic and cannot qualify a live backend.
  Original models, hashes, goldens and rtol=1e-3 / atol=1e-4 are unchanged.
- Isolation reports distinguish failed validation from successful cleanup and
  cooldown. Each experiment stops app/test processes, checks camera release,
  removes its forwards and waits for 30 continuous NORMAL seconds at <=32 C.
- After the first emulator thermal run, host-only tooling was corrected to reject
  hot devices before launching workers, retry force-stop for late worker startup
  and check the actual thermal error. The failed run remains recorded. These
  changes do not alter either APK; [before/after hashes and original build tools](candidate35-host-tooling-update.json)
  preserve provenance. The updated focused host suite passes 100 tests.

Vehicle TX remains compiled out and physical engagement unconditionally disabled.
Panda is disconnected, the phone is unmounted, and lens-cover trials are deferred.
No simulator fixture can qualify those physical gates. Candidate34 was built and
host-tested, then superseded before either device installation; its evidence is
retained separately.
