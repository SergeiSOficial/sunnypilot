#!/bin/sh
set -eu
cd /Users/sergios/sunnypilot
export SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk
export CXX=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++
.venv/bin/python -m unittest discover -s android/tests
.venv/bin/python android/reports/remaining-eight-20261104/check_projection_prefix.py
.venv/bin/python android/reports/remaining-eight-20261104/check_projection_shader.py convolution_packed_1x1s1d1
.venv/bin/python android/scripts/build_ncnn.py
.venv/bin/python android/scripts/prepare_accel_models.py --engine ncnn --model dmonitoring_model --ncnn-explicit-casts --ncnn-half-gemv
env JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home FOLD3_BUILD_PYTHON=/Users/sergios/sunnypilot/.venv/bin/python android/gradlew -p android --no-daemon --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
# Preserve archived artifacts: choose a NEW candidate name for a subsequent build.
# The following runners validate the already archived candidate45zo. Do not run concurrently.
.venv/bin/python android/reports/remaining-eight-20261104/run_prefix_full.py
.venv/bin/python android/reports/remaining-eight-20261104/run_prefix_trace.py
.venv/bin/python android/scripts/compare_ncnn_trace.py android/reports/remaining-eight-20261104/prefix-heads-trace-seed42.json
.venv/bin/python android/scripts/compare_ncnn_same_input.py android/reports/remaining-eight-20261104/prefix-heads-trace-seed42.json
.venv/bin/python android/reports/remaining-eight-20261104/run_controls_ab.py
# run_prefix_trace.py is an additional diagnostic and was not run on candidate45zo.
# run_emulator.py consumes the pending experiment inventory; preserve completed reports.
.venv/bin/python android/reports/remaining-eight-20261104/run_emulator.py
