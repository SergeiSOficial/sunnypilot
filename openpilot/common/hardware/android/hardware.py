"""Android hardware access goes through the application's public Java APIs."""
import os
from openpilot.common.hardware.base import HardwareBase


class HardwareAndroid(HardwareBase):
  def get_device_type(self):
    return "android"

  def get_serial(self):
    # Device identifiers are not accessible to an ordinary Android application.
    return "android"

  def get_os_version(self):
    from java import jclass
    return str(jclass("android.os.Build$VERSION").RELEASE)

  def get_current_power_draw(self):
    from java import jclass
    return float(jclass("ai.sunnypilot.fold3.AndroidHardware").powerWatts())

  def get_network_type(self):
    from openpilot.cereal import log
    return log.DeviceState.NetworkType.none

  def reboot(self, reason=None):
    raise PermissionError("APK lifecycle is managed by Android")

  def uninstall(self):
    raise PermissionError("Use Android's application manager")

  def get_cmdline(self):
    return {"android_data_root": os.environ["ANDROID_DATA_ROOT"]}
