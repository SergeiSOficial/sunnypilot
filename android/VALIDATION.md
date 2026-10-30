# Fold 3 port validation — 2026-10-30

**Overall: blocked. This is an Android qualification foundation, not the full
sunnypilot driving port requested in the plan. The APK cannot engage or send CAN.**
The connected phone retained stock Android 15, locked boot state and enforcing
SELinux. No Panda firmware was flashed and no vehicle settings were changed.

The new branch starts at the requested master commit
`6f7a2c94c10dffcf9a83b120ab1e6985e4496868`. All seven submodules and the original
ONNX model assets were populated at their pins. Native changes are reproducible
patches/recipes in this repository rather than uncommitted submodule edits.

## Acceptance matrix

`Passed` means the stated check ran successfully. `Failed` means a check ran
and did not meet its criterion. `Blocked` means missing implementation,
dependency or hardware prevented the required check. A passing component check
does not promote a blocked end-to-end requirement.

| Requirement | Status | Evidence or blocker |
|---|---|---|
| Requested branch and pinned sources | Passed | Git base and source pins |
| ARM64 API 35 APK build/install | Passed | Gradle, NDK and Chaquopy; physical phone and dedicated emulator |
| Python 3.12 on stock Fold 3 | Passed | Runtime report, actual embedded Python 3.12.15 |
| Native msgq/VisionIPC cross compilation | Passed | Android headers; no host extension binaries |
| Native Cap'n Proto/Cereal cross compilation | Passed | pycapnp 2.1.0, Cap'n Proto 1.0.1; extracted schema assets |
| App-private IPC layout | Passed component checks | Private paths, file permissions, memfd/SCM_RIGHTS; 100-frame native roundtrip |
| Cross-service upstream message graph | Blocked | No supervised driving daemons; native roundtrip is in one process |
| Complete upstream dependency set | Blocked | NumPy version mismatch; missing zstandard, pyzmq, Params/CAN/UI native libraries and other runtime dependencies |
| Android process supervision/restart | Blocked | Separate diagnostic service processes exist; upstream manager/restart policies are not ported |
| Android hardware backend | Blocked | IMU sampling, battery, thermal, storage and capability probes only; no Cereal sensor/GNSS/audio backend |
| Camera inventory/session matrix | Passed diagnostic | 26 actual attempts, characteristics and failures retained |
| Simultaneous rear wide + ultrawide | Failed | Independent 0+2 rejected; logical 20/23 unavailable to app |
| Two rear cameras + driver | Failed | Second rear sensor rejected |
| Road + front camera capture | Passed short diagnostic | Distinct camera IDs and overlapping delivered timestamps |
| Sustained road + driver camera capture | Failed | Both requested 30-minute attempts failed early |
| Camera-to-VisionIPC driving bridge | Blocked | Conversion and IPC tested separately; calibration/publication absent |
| Calibrated RHD driver monitoring | Blocked | Capture is not DM; model parity failed; camera/mount calibration absent |
| OpenCL backend availability | Passed on phone | Native query and tinygrad arithmetic; unavailable on emulator |
| Original model numerical parity | Failed | Both models exceed synthetic-case tolerances |
| Sustained 20 Hz inference | Failed baseline | p99 98.319 ms DM / 74.858 ms driving, each run separately |
| 100 Hz controls / p99 <10 ms | Blocked | Android control stack not running |
| Raylib Android UI / 30 fps | Blocked | Current UI is diagnostic Android widgets |
| Cover/inner display and surface recovery | Blocked full requirement | Activity recreation tested; no Raylib/driving-surface lifecycle |
| Android USB permission and libusb descriptor route | Passed build/negative test | UsbManager grant -> authorized descriptor; invalid descriptor rejected |
| Actual red Panda identity / compatible firmware | Blocked | No Panda enumerated on phone; signature qualification not implemented |
| Receive-only Panda CAN, bus mapping, ignition, heartbeat | Blocked | No connected Panda/vehicle; transport is identity reads only |
| Physical Hyundai C harness / fingerprint | Blocked | No vehicle or harness available to inspect |
| Factory ACC/braking/AEB behavior | Blocked physical validation | No CAN actuation path; pinned Hyundai implementation unchanged |
| Hyundai/Panda host safety checks | Passed | 1701 passed, 198 skipped; no safety limits changed |
| Emulator installation and native component tests | Passed | 7 instrumentation checks on dedicated ARM64 Android 15 AVD |
| Physical Fold 3 native component tests | Passed | Same 7 instrumentation checks on the connected phone |
| Injected camera/sensor/CAN and daemon fault recovery | Blocked | Injection adapters and supervised runtime absent |
| Debug ADB MetaDrive closed-loop bridge | Blocked | Not implemented; no simulator-to-vehicle path exists |
| 30-minute combined workload/watchdog qualification | Blocked | Driving runtime absent; camera-only soak failed |
| Physical vehicle/road readiness | Blocked | All camera/model/runtime/Panda/vehicle gates remain unmet |

## Implemented components

- Gradle ARM64 app, pinned NDK/CMake recipes and Python 3.12 runtime; model
  reference generator with SHA-256 verification; bundled third party notices.
- Camera2 inventory, independent/logical multi-camera attempts, actual image
  delivery statistics and failure reporting. Reusable NV12 buffers support
  padded/interleaved planes; timestamp/sample storage and reader queues are bounded.
- Android msgq path adaptation and VisionIPC memfd allocation. Native tests
  preserve frame bytes, dimensions/strides, IDs, start/end timestamps and an
  explicitly false validity flag. Cereal schemas and Hyundai values are original.
- pycapnp/Cap'n Proto cross compilation. Android needs a root-directory O_PATH
  fallback for schema traversal, an app-local working directory and direct
  extracted schema paths instead of temporary resource copies.
- Basic hardware observations and separate runtime/model diagnostic services;
  activity recreation tests. These services do not implement daemon supervision.
- UsbManager permission handling and libusb wrapping of the authorized descriptor.
  Only Panda identity/version vendor IN reads are exposed. Firmware compatibility,
  safe CAN reception and watchdog behavior still require implementation and hardware.
- Actual original-model OpenCL compilation, host-reference comparison, warmup,
  work-time/deadline measurement and independently selectable tuning experiments.

## Tests run

Host command evidence is recorded in [reports/host-tests.txt](reports/host-tests.txt).
The minimal SCons build passed. Focused Hyundai + msgq/VisionIPC tests ran 22
tests with 2 skipped: **20 passed**. Hyundai safety ran 1899 with 198 skipped:
**1701 passed**. These exercise upstream host code, not physical Panda hardware.

JVM tests cover bounded 10,000-frame statistics, nonmonotonic timestamps and
nearest timestamp pairing. Android instrumentation covers padded/interleaved
YUV, invalid sizes/strides, native 100-frame VisionIPC delivery, invalid USB
descriptor handling, immutable transmission disablement, Cereal serialization
over msgq and three activity recreations. All **7 instrumentation tests passed
on each device**; both reports are committed. Activity recreation does not establish
fold/unfold behavior, service crash recovery or sensor/camera continuity.

The emulator is `sunnypilot_fold3_api35`, ARM64 Android 15/Google APIs,
`emulator-5560`, hardware acceleration with software graphics. There is no
OpenCL device in that emulator. Its tests cannot establish Fold 3 camera or
thermal behavior.

The native source-preparation recipe and APK build were rerun successfully after
the final schema patches. [build-validation.json](reports/build-validation.json)
records the successful build, two passing JVM tests and Python lint/syntax checks. Ripwire's
`quality_delta` was attempted but returned an environment error (it could not
find Git HEAD or a baseline); no quality-delta result is claimed.

[artifacts.json](reports/artifacts.json) records the final APK and test APK
hashes. Both installed packages matched those hashes on the phone and emulator;
see [installed-artifacts.json](reports/installed-artifacts.json). The final
phone instrumentation rerun passed all seven tests in 2.899 s.

## Physical failures and integration gates

See [CAMERAS.md](CAMERAS.md) and [PERFORMANCE.md](PERFORMANCE.md), with raw reports
under [reports/](reports/). Both model runs and camera soaks were physical-phone
tests. The combined workload, closed-loop simulator and vehicle checks did not
run. Thermal status remained normal during sampled camera-only intervals; the
root cause of capture failures is not established.

The optional kernel-search experiment was stopped after 1,092.266 s while still
compiling its first model. Its partial state and interruption reason are retained;
it supplies no timing, parity or optimization result. This interruption is not
classified as a measured model failure.

For the 2019 Ioniq PHEV, preserve the pinned
[`HYUNDAI_IONIQ_PHEV_2019` implementation](https://github.com/sunnypilot/opendbc/blob/aa9bff6f5442cef8cb8e25e143ef38865c276ce8/opendbc/car/hyundai/values.py),
factory ACC/braking/AEB and approximately 32 mph (51.5 km/h) steering minimum.
Experimental longitudinal remains unavailable. No torque, steering minimum,
disengagement, liveness or Panda safety checks were relaxed in this branch.
Physical C-harness routing, actual fingerprint and firmware must still be verified.

Before any future actuation implementation, all of these gates must pass:
required distinct camera streams, calibrated RHD monitoring, valid calibration,
model parity/replay and sustained deadlines, upstream process/input watchdogs,
Panda firmware compatibility/disconnect/heartbeat, receive-only vehicle checks
and the unchanged Hyundai safety limits. The present APK's unconditional TX
disablement keeps engagement impossible while those checks are outstanding.

## Remaining implementation

The full plan still requires Android-compatible NumPy 2 and all native runtime
dependencies; Params and hardware backends; upstream daemon hosting/supervision;
calibrated camera/VisionIPC feeds and DM; a Raylib Android lifecycle backend;
Panda receive transport with firmware qualification and a tested simulator-only
ADB MetaDrive bridge. Required camera combinations and model parity/deadlines
must be resolved before physical actuation can be considered. This report does
not classify that remaining work as completed porting or performance optimization.
