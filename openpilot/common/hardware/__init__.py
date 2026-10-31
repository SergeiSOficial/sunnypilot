import os
import sys
from typing import cast

from openpilot.common.hardware.base import HardwareBase
from openpilot.common.hardware.comma.hardware import HardwareComma
from openpilot.common.hardware.pc.hardware import HardwarePc

AGNOS = os.path.isfile('/AGNOS')
ANDROID = hasattr(sys, "getandroidapilevel")
COMMA_HARDWARE = AGNOS
PC = not COMMA_HARDWARE


if ANDROID:
  from openpilot.common.hardware.android.hardware import HardwareAndroid
  HARDWARE = cast(HardwareBase, HardwareAndroid())
elif COMMA_HARDWARE:
  HARDWARE = cast(HardwareBase, HardwareComma())
else:
  HARDWARE = cast(HardwareBase, HardwarePc())
