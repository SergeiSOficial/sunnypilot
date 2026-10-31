# Fold 3 port validation

**Overall: BLOCKED.** The Android runtime and diagnostic visual pipeline are
implemented, but camera concurrency, original-model parity, calibration and
physical vehicle qualification prevent a complete driving port. Transmission
is compiled out and engagement is unconditionally disabled.

Target: stock Android 15 SM-F926B, Snapdragon 888 / Adreno 660; right-hand-drive
2019 Hyundai Ioniq PHEV, factory ACC, red Panda. Branch
`codex/fold3-android-port` starts at
`6f7a2c94c10dffcf9a83b120ab1e6985e4496868`. Pinned submodules are unchanged.
Android and Panda firmware have not been replaced or flashed. The user confirmed
that the Panda is disconnected.

## Acceptance

| Area | Result | Evidence and limit |
|---|---|---|
| ARM64 build / installation | Passed component checks | Python 3.12, Android Bionic native modules, packaged model assets |
| IPC, frame conversion and parsers | Passed component checks | Stride/pixel validation, all NV12 rotations, native VisionIPC, original model output slices into invalid upstream Cereal messages |
| Host runtime / input guards | Passed | [72 tests](reports/host-runtime-vision-tests.txt), including timestamp pairing, missing ultrawide and corrupt/unsafe Panda inputs |
| Fold 3 / emulator instrumentation | Passed | 14 tests on each device; raw reports below |
| Raylib lifecycle / display recovery | Passed component checks | Visible TX-disabled banner and preserved worker PIDs; physical fold transition remains untested |
| MetaDrive steering / factory ACC fixture | Passed component checks | Phone and emulator reruns; synthetic perception, calibration and torque-to-angle plant |
| Wide + ultrawide / two rear + driver | Failed | [Fresh Camera2 matrix](reports/fold3-camera-vision.json): stock HAL rejects second rear with MAX_CAMERAS_IN_USE |
| Road + driver preview, 60 seconds | Passed diagnostic | [Fold 3 live run](reports/fold3-live-vision.json): independent streams, no recovery, fresh previews; every input remains invalid |
| Delivered APK, 60-second live rerun | Failed | [Final Fold 3 run](reports/fold3-live-vision-final.json): road camera disconnected, recovered after three attempts; DM published zero predictions |
| Sustained combined diagnostic | Failed | [30-minute attempt](reports/fold3-live-vision-30min-summary.json): critical thermal status at 20.8 minutes; operator stopped the workload; zero accepted DM predictions |
| Critical thermal camera abort | Passed emulator component check | [Startup, active capture, host cleanup and new-activity stop cases](reports/emulator-thermal-abort.json): no thermal retry, override reset; physical thermal protection retained |
| Emulator live camera/model workload | Failed | Software renderer/cameras miss throughput target and report future sensor timestamps; fixture instrumentation still passes |
| Original driving / DM numerical parity | Failed | CPU, NNAPI and OpenCL reports in [PERFORMANCE.md](PERFORMANCE.md); no precision/tolerance change accepted |
| Sustained 20 Hz perception | Failed | OpenCL p99 55.7 ms driving / 82.8 ms DM; isolated CPU DM timing passes, parity does not |
| Full 100 Hz controls | Blocked | Measured work subset passes in short simulations; physical input chain and complete-loop deadlines are unqualified |
| Panda receive-only transport | Passed software / blocked physical | Pinned decoder, USB IN-only bindings and negative descriptor tests; no hardware connected |
| Firmware, harness, ignition and vehicle fingerprint | Blocked | Requires actual Panda + vehicle; signature recording is not firmware approval |
| Calibration / usable driver monitoring | Blocked | Preview affine and phone mounting transforms are unqualified; late predictions are discarded |
| Full combined driving acceptance / road use | Blocked | Missing rear stream, numerical failures and physical qualification cannot be established by simulator results |

## Implemented runtime

The manager supervises separate Controls, Monitoring, Sensors, RoadModel and
DriverModel Android service processes. Physical controls wait for real vehicle
parameters. Road inference waits for distinct rear sensors 0 and 2. Driver
inference consumes timestamp-gated camera 1 frames using the original model,
reusable projection buffers and upstream Cereal parsers. All messages remain
invalid; future, duplicate, stale, unsynchronized and non-finite inputs/outputs
are rejected. Non-finite predictions clear recurrent model state. The uncalibrated
preview transform is explicitly excluded from driving qualification.

Camera2 publishes bounded NV12 rings with sensor metadata, orientation and
BOOTTIME-to-MONOTONIC timestamp conversion. A cross-process ownership lock keeps
camera diagnostics and live capture from competing. Startup staleness begins
only after sessions are configured. Capture recovery is bounded and remains a
qualification failure. Critical thermal status rejects startup or closes active
capture within the one-second polling interval, with no retry. The UI displays only frames aged 0..100 ms and clears
stale previews. Closing/recreating the UI preserves the driving services.

The native ONNX wrapper checks the entire input set, shape, dtype and contiguous
buffer length, releases the GIL for inference and copies typed outputs. Model
benchmarks run as a foreground service. The CPU four-thread diagnostic profile
measured lower isolated DM work time but fails original-reference parity. No profile
is approved for driving.

The Panda path obtains an Android-authorized descriptor, verifies VID/PID and
red hardware, checks exact pinned CAN/health versions, then requires silent
safety mode, clear controls flags and no faults before receiving CAN. Repeated
health checks, fragmented-packet checksums, disconnect handling and bounded bus
fingerprints are implemented. CAN messages remain invalid and use host USB
arrival time, because the protocol has no sensor timestamp. Only whitelisted
vendor IN reads and bulk endpoint 0x81 exist. No USB OUT, safety-mode change,
heartbeat, reset or flashing API is exposed. Signatures are recorded without
claiming approval. [Phone](reports/fold3-panda-receive-only.json) and
[emulator](reports/emulator-panda-receive-only.json) absence checks report blocked.

## Evidence and interpretation

Final instrumentation: [Fold 3](reports/RFCR81Y1TBJ-vision-instrumentation.txt),
[emulator](reports/emulator-5560-vision-instrumentation.txt). Four perception tests
extend the original ten with NV12 rotation, native model-wrapper validation,
pinned Panda decoding/health checks and real upstream model-output serialization.
The focused host suite runs 72 tests. Earlier unchanged-pin checks passed 20
Hyundai/IPC tests (2 skipped) and 1,701 Panda safety tests (198 skipped); these
are software checks, not hardware or road qualification.

The fresh Fold 3 60-second live diagnostic delivered 1,792 road frames at 29.87
fps and 1,788 driver frames at 29.83 fps, with no capture recovery. Nearest stream
skew p99 was 16.59 ms. Raylib averaged 29.31 fps. DM computed 1,489 frames, published
zero and rejected every prediction as late; work p99 was 57.22 ms. Road inference
correctly remained blocked on missing sensor 2. Passing this diagnostic does not
mean valid perception, calibration, driver monitoring or vehicle control.

The dedicated API 35 ARM64 emulator uses SwiftShader and emulated front/rear
cameras. The first live run encountered a System UI ANR and stopped advancing
the UI. Closing the ANR restored rendering. The [full-size rerun](reports/emulator-live-vision-rerun.json)
still failed combined capture throughput, with ~14 fps cameras and ~11.8 fps UI;
[1280x720](reports/emulator-live-vision-1280.json) also failed capture throughput.
Many emulated camera timestamps were slightly future; the freshness guards
rejected them. Neither guard nor the 100 ms simulator deadline was relaxed.

Simulation uses separate Params/IPC namespaces and an ADB-UID restricted socket
with a strict service allowlist. It cannot relay CAN or Panda messages. Vehicle
transmission is false in every response. Factory ACC is preserved; experimental
longitudinal control is disabled, and the pinned steering minimum remains
14.30528 m/s (~51.5 km/h). The fixture's synthetic perception/DM/calibration and
simple torque-to-angle plant do not model Hyundai EPS or prove a visual closed
loop. A loaded fault run that disconnected is retained alongside isolated reruns.

See [CAMERAS.md](CAMERAS.md) for sustained camera results and
[PERFORMANCE.md](PERFORMANCE.md) for the measured profiles and rejected precision
experiments. Full localization, calibrated model projection, planning, physical
car discovery and vehicle qualification still require implementation/validation.

## Retained development failures

The `*-instrumentation-overlap.json` phone files are invalid harness runs: the
previous instrumentation teardown stopped the app during validation. The
`fold3-live-vision-startup-stale-report` files record a harness failure caused
by an old persisted manager report. The validator now requires a report from
the current launch. These runs are excluded from acceptance.

The short/initial live files retain the zero-buffer VisionIPC handshake and
early camera-startup failures fixed before the final APK. The loaded ADB fault
run, original-model numerical failures, Samsung session rejections and emulator
throughput failures remain failed results; isolated reruns do not erase them.

The [15-second phone smoke test](reports/fold3-live-vision-15s-preview-failure.json)
captured 450 road and 434 driver frames without recovery, but its sampled UI
observed only one fresh driver preview, below the validator's requirement.
That failed result is retained separately from the longer final smoke test.
The first final smoke launch also exposed a recent persisted simulator-manager
report during restart; [that harness failure](reports/fold3-live-vision-final-stale-manager.json)
is excluded. Validation now requires a new manager PID and matching sensor PID.

The final 60-second APK rerun failed on `camera_disconnected:0`; two recovery
attempts briefly found camera 0 unavailable, then the last session delivered
1,415 road / 1,417 driver frames near 30 fps. Recovery remains a qualification
failure. DM computed 1,115 predictions, discarded all as late and measured
57.74 ms work p99. Raylib averaged 29.29 fps. No further success is inferred
from the recovered session.

The compact APK removes 207,896,602 bytes of unused incremental ZIP space
(44.85%). All 74 packaged entries have identical content hashes to the
precompact development APK; see [packaging comparison](reports/vision-apk-packaging-comparison.json).
That comparison predates the final thermal-guard change. The delivered APK was
rebuilt and tested separately; its identity and limits are recorded below.
Build pins, APK sizes and hashes are in [artifacts.json](reports/artifacts.json);
[installed-artifacts.json](reports/installed-artifacts.json) records device hashes.
