# Fold 3 port validation

**Overall status: BLOCKED — partial Android runtime implemented; full visual
driving and vehicle qualification are incomplete.** Vehicle transmission is
unconditionally compiled out. Neither simulator nor emulator success permits
physical engagement.

Target: stock Android 15 SM-F926B, Snapdragon 888 / Adreno 660; RHD 2019 Hyundai
Ioniq PHEV, factory ACC, red Panda. All pinned submodules stay at the inspected
base revisions. The APK is installed without changing Android or Panda firmware.

## Acceptance matrix

| Area | Status | Evidence / limit |
|---|---|---|
| ARM64 APK, stock Android installation | **Passed** | Native Bionic libraries, Python 3.12; final artifacts/hashes in reports |
| Cereal, Params, msgq and VisionIPC | **Passed** component checks | Java→native→Python frames plus serialization/shared-memory tests |
| Android service supervision | **Passed** component checks | Real controlsd process killed/restarted; manager and UI Activity recreation survive |
| controlsd and dmonitoringd | **Passed** runtime/import checks | Simulator drives real controlsd; physical controls/DM wait for qualified inputs |
| Raylib upstream landscape UI | **Passed** offroad rendering | Separate UI process and Android settings; visible lifecycle checks pass; onroad camera overlays unqualified |
| Android IMU/storage/thermal/battery | **Passed** basic acquisition | Raw IMU deliberately invalid: vehicle mounting calibration absent |
| GNSS and audio alerts | **Blocked** full integration | Public last-known-location/PCM APIs present; no validated live GNSS/alert daemon |
| Distinct dual rear capture | **Failed** | Public Camera2 rejects rear 0+2 at both tested sizes |
| Road + driver capture | **Passed** short only; **failed** sustained | Latest 30-minute request stopped at 663.268 s, stale front stream |
| Two rear + driver | **Failed** | Second rear camera rejected |
| Calibrated RHD DM and synchronized camera inputs | **Blocked** | All physical VisionIPC frames `valid=false`; no accepted transforms/input pairer |
| Original-model numerical parity | **Failed** | Host CPU and phone OpenCL discrepancies against original FP16 reference |
| FP32 own-reference parity | **Passed** synthetic only | Does not establish equivalence to original FP16 model or replay quality |
| 20 Hz inference / p99 <50 ms | **Failed** | Best original driving p99 74.858 ms; FP32 slower |
| Controls p99 work <10 ms | **Passed** measured subset only; full loop **blocked** | Phone 6.501 ms / 2,991 iterations, one ≥10 ms sample; reception/pose calibration and reporting excluded |
| MetaDrive→Android steering | **Passed** fixture | Actual world, real phone controller; synthetic perception and torque-to-angle plant |
| Factory ACC / steering minimum / cancellation / stale inputs | **Passed** simulator checks | No phone longitudinal activity or lateral activity below pinned minimum |
| Simulator CAN isolation | **Passed** protocol/component checks | Debug-only, app/ADB UID, isolated Params/IPC; CAN/Panda/output injection rejected |
| Panda USB | **Blocked** physical qualification | UsbManager→libusb identity IN only; no connected Panda |
| Hyundai safety | **Passed** host checks | Physical firmware, fingerprint, C harness, bus mapping, ignition/heartbeat unverified |
| 30-minute combined workload | **Blocked** | Complete visual stack absent and camera-only soaks fail |
| Physical car / road readiness | **Blocked** | No Panda/vehicle validation; TX disabled |

## Implemented runtime

- ARM64 cross-compilation of Params, pycapnp/Cereal, msgq/VisionIPC, libusb,
  pyzmq/libzmq, zstandard, setproctitle and shared Raylib/pyray. Native source
  exports and patches are reproducible in the main repository.
- Android foreground manager and separate Python worker services. Binder death,
  progress checks and bounded restart/backoff replace fork/multiprocessing.
  Controls, monitoring, raw IMU and debug simulator entry points are hosted.
- App-private Params, shared memory and sockets. The simulator has separate
  directories and no CAN/Panda message route. Existing Cereal schemas,
  controller algorithms and parameter types are retained.
- Real Camera2 NV12→VisionIPC delivery with bounded rings and actual timestamps;
  diagnostic calibration validity remains false. Metadata-only capture failures
  can be observed without immediately aborting an otherwise delivered image.
- NativeActivity Raylib UI and Android settings. GLES shaders declare the
  version first, which is required by the tested emulator compiler. Surface
  changes preserve GL resources and driving workers; Activity completion gets
  a fresh UI interpreter/renderer. The foreground driving Activity keeps the
  screen awake using Android's window flag.
- EGL configuration selection now checks for a compatible window configuration,
  retries without unsupported MSAA and rejects failed surfaces. The emulator's
  earlier black surface advanced frame counters, exposing a false-positive
  lifecycle check. Final lifecycle checks require a visible transmission banner
  and both the surface and screenshot reaching the requested landscape size.
- Debug MetaDrive/Cereal relay with finite input ages, ordered bounded packets,
  typed allowlists and disconnect/cancel/stale withdrawal. Factory ACC remains
  simulated by the host. The pinned steering minimum is 14.30528 m/s (51.5 km/h).

## Test evidence

The Android instrumentation suite contains **10 tests per device**. It covers
NV12 padding/interleaving and invalid layouts, native VisionIPC, Java→Python
changing frame delivery, Cereal/msgq, native Params, immutable TX disablement,
USB invalid descriptors, runtime/UI imports and process restart supervision.
Final raw results and lifecycle reports are under [reports/](reports/).
The delivered APK passes 10 tests on the Fold 3 (8.985 s) and emulator
(15.456 s), plus two JVM tests. Installed app and test APK hashes match on both
devices. Lifecycle and ADB fault reports identify this exact delivered APK.
MetaDrive and UI/control timing reports identify the preceding build; its only
delivery difference is the final Activity screen-awake flag. Camera/model
qualification retains the earlier development measurements and their failures.
The emulator is dedicated ARM64 Android 15 with software graphics and 4 GiB RAM;
it does not expose the phone's OpenCL driver or camera combinations.

After validation the Fold 3 was restored to physical diagnostic mode. Its
[manager snapshot](reports/fold3-physical-manager-final.json) has Controls,
Monitoring and Sensors, no Simulator, and engagement/TX disabled. Controls wait
for physical vehicle parameters; monitoring waits for driver-model output. The
[IMU snapshot](reports/fold3-physical-imu-final.json) reports uncalibrated,
invalid samples. This startup check does not establish sustained performance.

[Host runtime tests](reports/host-runtime-tests.txt) cover simulator framing,
CAN/output injection rejection, sequence/age limits, steering-minimum fixtures,
Cereal SubMaster list/data/text initialization, Params, longitudinal/lateral
controllers and driver monitoring. The earlier minimal SCons build passed.
The final focused runtime suite passes 58 tests. ADB fault validation rejects
ten malformed/disallowed session types and checks missing-model inactivity,
recovery and abrupt disconnect. The initial missing-model test exposed an
active/inactive publication race, retained in
[the failed regression run](reports/emulator-adb-faults-before-fix.json).
The fix replaces active states before publishing an invalid packet's inputs.
The host namespace guard is also regression tested against a misspelled
environment variable; both fixtures use msgq's actual `OPENPILOT_PREFIX`.
The final emulator fault run passes with software rendering backgrounded.
A loaded run with foreground rendering and concurrent host MetaDrive hit the
unchanged 100 ms socket timeout after 29 accepted packets; it disconnected with
no worker restart. [That failure is retained](reports/emulator-adb-faults-rendering-timeout.json).
It prevents a combined emulator scheduling qualification; the timeout was not
relaxed to make the isolated rerun pass.
Earlier focused Hyundai/msgq/VisionIPC ran 22 tests: 20 passed, 2 skipped.
Hyundai safety ran 1,899: 1,701 passed, 198 skipped. Those checks were host-side,
not Panda firmware execution or actual vehicle integration.

[MetaDrive emulator cancellation](reports/emulator-metadrive-cancel.json) and
[Fold 3 stale-input run](reports/fold3-metadrive-stale.json) ran 30-second real
worlds with the phone controller. Perception, calibration and driver output are
synthetic; the plant is a damped torque-to-angle fixture, not a Hyundai EPS model.
The final emulator MetaDrive run also backgrounds software rendering.
No full camera-to-control visual loop is claimed. The validation world uses
multiple 60 m straight blocks: the initial single long-block fixture ended
out-of-road even with zero steering and was replaced before successful runs.

## Remaining implementation and gates

The full supervised modeld/driver-model, localization, calibration, planner and
selfdrived pipeline is not integrated. Native estimation/model preprocessing
extensions and Android NumPy 2 remain outstanding. GNSS freshness and alert
daemon integration, calibrated mounting/orientation, synchronized valid camera
feeds and replay-equivalent models are not implemented/qualified. Public Camera2
concurrency and long-running front capture are physical blockers, and the measured
OpenCL profiles cannot meet the 20 Hz target.

USB transport currently performs identity reads only. Compatible firmware,
receive-only CAN, Panda heartbeat/disconnect handling, actual vehicle fingerprint,
bus/harness routing and all physical vehicle checks remain required. No flashing
or transmission path was added. Factory ACC, braking/AEB, approximately 51.5 km/h
steering minimum, torque limits and disengagement/safety checks are retained from
[pinned Hyundai code](https://github.com/sunnypilot/opendbc/blob/aa9bff6f5442cef8cb8e25e143ef38865c276ce8/opendbc/car/hyundai/values.py).
Experimental longitudinal remains disabled. All these gates must pass before a
future actuation implementation can be considered; this APK cannot transmit.
