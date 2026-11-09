# Host launch blocker — 2026-11-04

No new runtime patch, APK or device test was completed during this continuation. Apple-provided Python 3.9.6 successfully ran 37 focused controls scheduling, reporting, CPU-placement and experiment-isolation tests; `stdlib-host-tests.log` preserves this fresh PASS. These mocked host checks do not establish measured cycle timing or full-model parity. Candidate45zo and its failed accuracy/timing evidence remain unchanged. TX remains disabled.

On macOS 27.2 (26B123), Python, both bundled and Homebrew ripgrep, and ADB stalled before application startup. Python's sampled footprint was 96 KiB, with its only sampled stack frame `_dyld_start`; `python-startup-sample.txt` preserves the sample. These launches remained stalled for over two minutes. Built-in Apple commands continued working. Concurrent observations showed syspolicyd at approximately 570–614 percent CPU. This suggests a host executable-launch problem; the precise cause has not been established.

Only task-owned Python/ripgrep/ADB client probes were interrupted. No security service, existing ADB server or unrelated process was stopped. No emulator was started and no forwarding was added. Host builds, numerical checks and emulator tests are blocked until executable launches recover. A Mac restart was requested from the user; no system/security change was performed.

After recovery, continue from candidate45zo: capture the earlier convolution prefix with existing bounded ncnn trace windows, compare original-reference and same-input sections, and identify the next accumulation/rounding divergence before editing kernels. Reuse the matched controls fixture and retain all raw complete-cycle/deadline counters. Do not promote a backend or advance sustained workloads while accuracy/timing gates fail.

A subsequent Python 3.12 launch succeeded, but importing ONNX failed while loading ml_dtypes with `library load mig callout failed` and code-signature validation errors. See `numerical-import.log`. ADB and ripgrep launches continued to stall. Executable startup therefore recovered only partially; numerical/emulator validation remains blocked. No libraries were re-signed and no security checks were disabled.

Focused checks reproducible independently of the affected native dependencies:
```sh
/usr/bin/python3 -m unittest android.tests.test_controls_scheduling android.tests.test_controls_reporting android.tests.test_controls_affinity android.tests.test_experiment -v
```

## Recovery observed later on 2026-11-04

After the user reported the phone connected, Python 3.12 numerical imports and ADB launches succeeded. ADB identified the SM-F926B as RFCR81Y1TBJ. No system security configuration was changed. Work resumed in `../fold-reconnected-20261104/`. The first complete host test run failed two compiler checks because it used the default SDK; rerunning with the established pinned Xcode 26.5 SDK passed all 190 tests. This supersedes the launch blocker for this continuation without altering earlier failed evidence.
