#!/bin/zsh
set -eu
cd /Users/sergios/sunnypilot
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
export CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
export FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python
.venv/bin/python android/scripts/build_ncnn.py
.venv/bin/python android/scripts/prepare_accel_models.py --engine ncnn --model dmonitoring_model --ncnn-explicit-casts --ncnn-half-gemv
.venv/bin/python -m unittest discover -s android/tests
/Users/sergios/Library/Android/sdk/cmake/3.31.6/bin/cmake -S android/build/native/accelerators/glslang-source/glslang-780b6a209132dc74e99458f62ebe5fa7e57c41cf -B android/build/native/glslang-host-check -DENABLE_GLSLANG_BINARIES=ON -DENABLE_OPT=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
/Users/sergios/Library/Android/sdk/cmake/3.31.6/bin/cmake --build android/build/native/glslang-host-check --target glslang-standalone -j4
.venv/bin/python android/reports/linear-timing-repair-20261103/check_linear_shader.py
.venv/bin/python android/reports/linear-timing-repair-20261103/check_linear_arithmetic.py
android/gradlew -p android --no-daemon --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
# Archive under a NEW candidate name; do not replace previous evidence.
# Emulator runner consumes the immutable candidate45zj archive and its reviewed
# inventory. Completed inventory entries are retained, not silently rerun.
# .venv/bin/python android/reports/linear-timing-repair-20261103/run_emulator_45zj.py

# After the functional runner has stopped its emulator and completed cleanup:
# .venv/bin/python android/reports/linear-timing-repair-20261103/run_full_ncnn_software.py
