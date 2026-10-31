# Fold 3 Samsung camera validation — 31 October 2026

Samsung's mode-37 session now operates both rear sensor pipelines with the
front camera on the stock SM-F926B. **Driving qualification remains blocked.**
The application still needs independently extracted, calibrated wide/ultrawide
images; no multiview frame is accepted as a driving-model input.

Branch: `codex/fold3-android-port`. Package: `com.samsung.sunnypilot.fold3`.
APK SHA-256: `0e05c9927a2a12755876ba539ac3eec7d2bcee1f32e89228d2e03c5498cc850a`.
[Build identity and source hashes](artifacts.json), [build log](build.txt),
[reproduction commands](../../README.md#samsung-multiview-diagnostic-build).
Vehicle transmission is compiled out; engagement remains unconditionally disabled.

| Check | Result | Evidence / limit |
|---|---|---|
| Compatibility APK / installation | Passed | Same APK hash in all five final experiment isolation records |
| Host Android tests | Passed | [29 tests](host-tests.txt), [Ruff](ruff.txt) |
| Host simulator protocol | Passed | [8 tests](host-simulator-tests.txt); no fresh MetaDrive closed-loop run claimed |
| JVM statistics | Passed | [2 unchanged cached tests](jvm-tests.xml) |
| Phone instrumentation | Passed | [16 tests](fold3-final-instrumentation.txt), including actual opaque-preview pixels/orientation |
| Emulator instrumentation | Passed | [16 tests](emulator-final-instrumentation.txt) |
| Three active hardware sensors | Passed | [HAL excerpt](fold3-three-camera-hal.txt), [structured evidence](fold3-three-camera-evidence.json): IMX258 ultrawide + IMX555 wide + IMX374 front |
| Samsung multiview + front transport | Passed | [60 seconds](fold3-final-samsung-front.json), no disconnect or provider restart |
| Independent rear driving images | Blocked | Composite/preview images and HAL activation do not establish separate calibrated model inputs |
| Ordinary phone rear/front capture | Passed capture only | [60 seconds](fold3-final-road-front.json), 1,807 / 1,799 frames at 29.99 / 30.02 fps |
| Ordinary emulator rear/front capture | Passed capture only | [10 seconds](emulator-final-road-front.json), 250 / 249 frames at 25.00 / 24.99 fps |
| Emulator driving-input timing | Failed | 239 / 238 future timestamps; nearest-skew p99 49.15 ms exceeds 20 ms. Inputs remain invalid |
| Physical Panda / vehicle | Blocked | Panda disconnected; no CAN/harness/firmware qualification |

The current three-camera transport delivered 1,800 rear YUV thumbnail
frames at 29.99 fps, 1,794 rear opaque-preview frames, and
1,805 front YUV frames at 30.08 fps. Rear/front timestamp and
packing errors were zero. Preview and thumbnail diagnostic samples shared
sensor timestamp `97193380131077` ns. All 1,807 logged rear activation
vectors in this trial were `[1:1:0]`; the third rear (telephoto) pipeline was inactive.

Rear/front packing p99 was 2.51 / 2.09 ms; delivery-age
p99 was 99.10 / 98.88 ms. There were
9 / 15 deliveries over 100 ms. This is a camera-only diagnostic,
not a sustained perception benchmark. Phone skin temperature was sampled at
27.9°C before and 30.7°C after; thermal status was 0 at both
samples. No claim is made about a new 30-minute run or combined model deadlines.

The Samsung trial intentionally retains `status: blocked` and the validator's
nonzero exit because physical-image qualification is incomplete. Its generic
[experiment wrapper](fold3-final-samsung-front-isolation.json) consequently says
`failed` for the child exit; `transport_status: passed` is a narrower result.
HAL evidence is kept separately from the app's stricter
`distinct_sensor_evidence: false` / `visionipc_valid: false` qualification fields.
The prior failed model parity and sustained perception results in
[VALIDATION.md](../../VALIDATION.md) remain failed and retain their original APK hashes.

Every final experiment force-stopped both application variants and their test
packages, waited at least 30 seconds with normal thermal state and phone skin
at most 32°C, and checked camera-provider PIDs and teardown afterward. Final
[phone cleanup](fold3-cleanup.json) records no application processes or owned
camera clients, thermal status 0 and skin 29.0°C.
The [emulator](emulator-cleanup.json) was cleaned and powered off. Private camera
samples were deleted from the phone and Mac after inspection; no images or
Samsung binaries are committed.

Earlier targeted failures are retained with their actual installed APK hashes.
Ordinary physical-output sessions for hidden IDs and mode 32 failed, including
provider restarts. Hidden IDs are excluded from the automatic matrix. The
retained emulator launch failure used an invalid ADB flag and was superseded
by the successful final instrumentation run. Other intermediate successful
reports describe earlier builds, not this delivery APK.
