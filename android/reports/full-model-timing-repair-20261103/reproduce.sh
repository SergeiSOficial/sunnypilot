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
.venv/bin/python android/reports/full-model-timing-repair-20261103/check_half_heads.py
.venv/bin/python android/reports/full-model-timing-repair-20261103/check_activation_mean.py
.venv/bin/python android/reports/full-model-timing-repair-20261103/check_rounding_shaders.py tanh
.venv/bin/python android/reports/full-model-timing-repair-20261103/check_rounding_shaders.py reduction
android/gradlew -p android --no-daemon --console plain -Pfold3CameraCompat=true -Pfold3Qnn=true -Pfold3MnnPatched=true -Pfold3NcnnPatched=true :app:testDebugUnitTest :app:assembleDebug :app:assembleDebugAndroidTest
# Archive under a NEW candidate name only after successful build; do not overwrite 45zk/45zl.
# Runners have fixed output names: copy them and update reports for independent repeats.
# Existing evidence runners, in sequence with no other host build/test during controls timing:
# .venv/bin/python android/reports/full-model-timing-repair-20261103/run_controls_ab.py
# .venv/bin/python android/reports/full-model-timing-repair-20261103/run_rounding_ncnn.py
# .venv/bin/python android/reports/full-model-timing-repair-20261103/run_emulator_45zl.py

# Final diagnostic trace, after functional runner cleanup:
# .venv/bin/python android/reports/full-model-timing-repair-20261103/run_rounding_heads_trace.py
# .venv/bin/python android/reports/full-model-timing-repair-20261103/check_rounding_gpu.py
# .venv/bin/python android/scripts/compare_ncnn_trace.py android/reports/full-model-timing-repair-20261103/rounding-heads-trace-seed42.json
