# Fold 3 port validation

**Overall: BLOCKED.** Vehicle transmission is compiled out and engagement remains
unconditionally disabled. The complete visual driving stack is not operational.
Independent rear images for the models, numerical parity, timely driver
monitoring, calibration and physical vehicle qualification remain unmet.
Samsung multiview transport with both rear pipelines and the front camera is
now operational in a separate diagnostic installation.

Target: stock Android 15 SM-F926B, Snapdragon 888 / Adreno 660; right-hand-drive
2019 Hyundai Ioniq PHEV, factory ACC and red Panda. Branch
`codex/fold3-android-port` starts at
`6f7a2c94c10dffcf9a83b120ab1e6985e4496868`. Pinned submodules, stock Android,
factory ACC/AEB configuration and Panda firmware are preserved. The Panda is
**disconnected**; no physical CAN, harness or firmware qualification is claimed.

## Delivery identity and test scope

Current Samsung compatibility debug APK (`com.samsung.sunnypilot.fold3`) SHA-256:
`0e05c9927a2a12755876ba539ac3eec7d2bcee1f32e89228d2e03c5498cc850a`.
The camera changes, fresh build, host/device checks and each installed hash are
recorded in the [camera validation report](reports/camera-compat/README.md). This build adds Samsung's
mode-37 multiview layout, opaque-preview readback and front capture. It retains
the unconditional engagement and vehicle-transmission blocks.

Previous regular-package debug APK SHA-256:
`e5347db807f6551886d7de706d6ebb939c51dcd25c5c0ade1cd57b7fb24941ee`.
[Installed hashes](reports/installed-artifacts.json) match on the Fold 3 and the
API 35 ARM64 emulator. Build commands are in [README.md](README.md); artifact
sizes, source pins and hashes are in [artifacts.json](reports/artifacts.json).

That APK uses two non-spinning model threads. Instrumentation, phone
60-second live capture, the phone soak attempt, emulator lifecycle/resize,
emulator live capture and thermal-abort tests exercise this APK. The immediately
preceding `31c9028b…` APK used four live threads. Its unchanged camera APIs,
individual model benchmark profiles, phone lifecycle, fault tests, MetaDrive and
Panda-absence checks are retained under [four-thread-validation](reports/four-thread-validation/installed-artifacts.json).
Each experiment's `*-isolation.json` records its actual installed APK hash.
These results are not silently attributed to the newer APK.

## Acceptance

| Area | Result | Evidence and limit |
|---|---|---|
| ARM64 APK / installation | Passed | Gradle build, native Bionic modules, signing and installed hashes verified |
| Focused host checks | Passed | 29 Android tests and 8 simulator protocol tests; [current results](reports/camera-compat/host-tests.txt), [protocol results](reports/camera-compat/host-simulator-tests.txt) |
| JVM tests | Passed | Two unchanged FrameStats tests in the compatibility build |
| Fold 3 / emulator instrumentation | Passed | 16 each, including opaque preview pixels/orientation; [phone](reports/camera-compat/fold3-final-instrumentation.txt), [emulator](reports/camera-compat/emulator-final-instrumentation.txt); each isolation report names its actual APK |
| Exact preprocessing / IPC | Passed component checks | Padded/rotated NV12, native gather equivalence, malformed buffers, VisionIPC and parser checks |
| Standard Camera2 matrix (prior build) | Partial | Phone 16/26 and emulator 6/12 attempts passed; no disconnects, all readers/devices closed; ordinary dual rear combinations failed |
| Samsung rear multiview + front transport | Passed | Mode 37, logical 20 + front 1; 60-second capture, provider remains stable; [camera details](CAMERAS.md) |
| Independent wide / ultrawide driving inputs | Blocked | Direct physical-output sessions fail; multiview thumbnails/opaque preview need independent extraction and calibration before VisionIPC/model acceptance |
| Standard rear/front capture regression | Passed capture only | [Phone 60 seconds](reports/camera-compat/fold3-final-road-front.json), [emulator 10 seconds](reports/camera-compat/emulator-final-road-front.json); virtual-camera timing does not qualify model inputs |
| Phone 60-second live diagnostic (previous regular APK) | Passed capture/preview only | [Previous run](reports/fold3-final-live.json): 1,800 rear / 1,794 front frames, no recovery, all DM predictions late |
| Phone 30-minute diagnostic attempt | Passed capture; failed perception | [Soak result](reports/fold3-final-soak.json); no full road/planning/control workload |
| Raylib lifecycle / emulator resize | Passed component checks | [Phone](reports/four-thread-validation/fold3-final-lifecycle.json), [emulator](reports/emulator-final-lifecycle.json); physical fold transition untested |
| Process fault recovery | Passed component checks | [Phone](reports/four-thread-validation/fold3-final-faults.json), [emulator](reports/four-thread-validation/emulator-final-faults.json) |
| MetaDrive / factory ACC / stale-input cancellation | Passed fixture checks | [Phone](reports/four-thread-validation/fold3-final-metadrive.json), [emulator](reports/four-thread-validation/emulator-final-metadrive.json); synthetic perception/DM/calibration, no physical TX |
| Critical thermal abort | Passed emulator component checks | [Final APK](reports/emulator-final-thermal.json); injected emulator thermal state reset afterward |
| Emulator live workload | Failed | [Current run](reports/emulator-final-live.json): ~13–14 fps cameras / ~12 fps UI; future sensor timestamps rejected |
| Original model parity | Failed | Three synthetic cases per model, original tolerance unchanged; finite fallback still differs |
| Sustained 20 Hz / p99 <50 ms perception | Failed | DM live work misses deadlines and all outputs are late; driving CPU exceeds 50 ms and native FP16 outputs are non-finite |
| Complete 100 Hz controls | Blocked | Only isolated/synthetic work has been measured; complete physical input and planning chain unqualified |
| Panda receive-only transport | Passed software / blocked physical | IN-only bindings, pinned decoder/health checks and negative descriptor tests; no Panda connected |
| Vehicle fingerprint / ignition / harness / firmware | Blocked | Real Panda and vehicle checks still required |
| Engagement / vehicle transmission | Disabled | Unconditional engagement block and no USB OUT transport exposed |

The 30-minute road/front capture and preview test **passed** with zero recoveries. It delivered 53,964 rear and 54,003 driver frames at 29.98/30.01 fps; both image-handler drains and camera-close callbacks completed. Final rolling nearest-skew p99 was 16.50 ms, and delivery-age p99 was 84.43/94.31 ms. The host observed 1,817.18 seconds including post-capture sampling.

**Sustained perception failed.** Android reached severe thermal status (3), with skin temperature sampled up to 44.6°C. DM computed 25,909 frames, all late, and published zero; rolling work p99 reached 82.57 ms with 19,786 cumulative 50 ms deadline misses. The UI averaged 29.96 fps. Memory after 60 seconds ranged from 916.1 to 947.4 MiB PSS. See the [soak summary](reports/fold3-final-soak-summary.json) and [raw samples](reports/fold3-final-soak.jsonl). This workload excludes road inference, localization, planning, active physical controls and Panda, so it cannot qualify the complete driving stack.

## Changes and safeguards

Camera teardown stops/aborts requests, marks streams closing, drains image
callbacks, waits for device-close callbacks and only then releases the native
publisher and ownership lock. Every matrix attempt reports both drain and close
completion. The activity waits for its actual landscape surface before starting
capture, fixing the observed initial portrait-surface race. Rotation changes
fail the session rather than silently applying the wrong transform. Supported
noise reduction, edge enhancement and stabilization are disabled for the record
request template. Vendor/provider disconnects remain qualification failures.

The runtime is rebuilt from pinned ONNX Runtime source without optional
KleidiAI/SME/SVE kernels. This removes the emulator's `RDSVL` illegal-instruction
crash. Portable exports omit host-specific NHWC operators; their manifests
retain failed numerical checks against the original goldens. Neither successful
loading nor finite output establishes parity. The original precision and
`rtol=1e-3`, `atol=1e-4` remain unchanged. No precision experiment is accepted.

Native row gathering reuses buffers and reduces preprocessing work. The model
loop caps starts at 20 Hz without catch-up bursts and uses two non-spinning CPU
threads after measured isolated CPU savings. Queues remain bounded; missing,
future, duplicate, >100 ms old or >20 ms skewed frames are rejected. Late and
non-finite predictions are discarded. Every visual diagnostic message remains
invalid; calibration is not replaced with invented transforms. Road inference
requires both distinct rear sensors and correctly waits for missing sensor 2.

The separate Android services supervise upstream controlsd/dmonitoringd,
Android IMU/GNSS/battery/thermal inputs and Raylib. The full localization,
calibration, planning and vehicle stack remains incomplete. APK NumPy 1.26.2
does not satisfy upstream NumPy >=2; only the exercised subset is validated.
Updates are APK installs. AGNOS/system-service/firmware-flashing paths are excluded.

Panda bindings only expose whitelisted vendor IN reads and bulk IN endpoint
0x81. They verify red hardware and pinned protocol versions, require silent
safety mode, clear controls flags and no faults, and reject malformed packets.
There is no USB OUT, heartbeat, reset, safety-mode change or flashing API.
RX CAN remains invalid and uses host USB arrival timestamps. RHD behavior,
factory ACC/AEB and the approximately 51 km/h steering minimum remain intact.
All upstream torque/disengagement/watchdog safeguards are retained.

## Isolation, retained failures and remaining qualification

Every new experiment stops the application and instrumentation package, checks
that processes and camera clients are gone, waits at least 30 seconds with
normal thermal state (phone skin <=32°C), and verifies cleanup afterward.
Application data/calibration and caches are preserved. Workers are not restored
between experiments. Thermal overrides are never used on the physical phone.
The emulator was left clean with its override reset and then powered off.

The earlier emulator lifecycle precondition failed while a System UI ANR dialog
paused rendering; [failure evidence](reports/emulator-systemui-anr.txt) and the
[failed isolation report](reports/four-thread-validation/emulator-final-lifecycle-isolation.json)
are retained. After closing that dialog, the previous regular APK lifecycle/resize test
passed. A fresh cold-boot check again required closing a System UI ANR dialog;
[checked-in launcher validation](reports/emulator-final-wrapper-validation.json)
then passed lifecycle/resize, but unattended startup remains failed. Its live throughput still failed. Camera providers were not restarted
to manufacture a clean result.

Earlier phone camera stalls, the `e2ee0027…` startup disconnect/recovery and the
20.8-minute critical-thermal attempt remain in their original reports. The
landscape startup fix's intermediate result is retained under
[pre-power-fix](reports/pre-power-fix/fold3-final-live.json). Earlier 72 runtime
checks, Hyundai/IPC 20 tests run / 2 skipped and Panda safety 1,701 run / 198
skipped are historical unchanged-pin evidence, not new runs in this iteration.
Emulator, fixture and diagnostic successes cannot qualify a real vehicle.
