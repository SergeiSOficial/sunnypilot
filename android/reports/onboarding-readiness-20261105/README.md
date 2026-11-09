# Fold 3 and emulator readiness rerun

Candidate45zy defers training-guide PNG decoding until the user enters the training screen. Candidate45zx loaded those images while rendering the terms screen, during full-chain startup. The change preserves the terms, consent and training flow and keeps the existing readiness threshold.

- Gradle assembled the app and instrumentation APKs. Its unit-test task was up to date. The onboarding dispatch smoke check and 50 focused host unit tests passed.
- App SHA-256 `0e70a87decb28b63f49d7948f433c559de91336828851701145275b32d8f0811`. Test APK SHA-256 `96876c828001aafe773f8d00589528522dce16709372ab77e134117186de6a6f`. Installed hashes matched on both devices.
- Fold 3 passed 28 instrumentation tests and both 30-second full-chain fixtures. Steady controls averaged 99.997 Hz, with 2.266 ms work p99 and 12.257 ms complete-cycle p99. There were zero scheduled deadline misses and 1,213 raw cycle misses. The complete-cycle timing failure remains open.
- Emulator passed 28 instrumentation tests. The steady fixture passed after reaching Raylib readiness in 34.73 seconds. Candidate45zx had timed out at 180 seconds. Host load was uncontrolled across builds, so this is a readiness result for candidate45zy.
- Road-model conversion and the combined 795-comparison gate remain blocked. This APK was exercised with synthetic fixture inputs. Its model parity and sustained workload were not rerun. Vehicle TX remains compiled out and engagement is disabled.

From the repository root, build using the existing native and model assets:

```sh
export FOLD3_BUILD_PYTHON="$PWD/.venv/bin/python3.12"
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export ANDROID_HOME=/Users/sergios/Library/Android/sdk
android/gradlew -p android --offline --no-daemon --console plain \
  -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true \
  :app:assembleDebug :app:assembleDebugAndroidTest :app:testDebugUnitTest
```

`run_devices.py` uses the archived candidate45zy APKs. Run a copy from a fresh sibling report directory to preserve these receipts. It installs and checks hashes, runs each fixture in isolation, verifies cleanup and shuts down its dedicated emulator.

The emulator fault fixture also passed, with Raylib ready in 38.34 seconds. All six experiments passed isolation checks. Final receipts confirm stopped app/test processes, released cameras, empty forward lists and over 30 continuous seconds at NORMAL thermal status within the skin-temperature limit. The dedicated emulator shut down.
