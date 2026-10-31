"""Use stock Android settings for services owned by the OS."""
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.list_view import button_item, text_item
from openpilot.system.ui.widgets.scroller_tici import Scroller


class AndroidSettings(Widget):
  def __init__(self, panel):
    super().__init__()
    from java import jclass
    hardware = jclass("ai.sunnypilot.fold3.AndroidHardware")
    if panel == "network":
      items = [button_item("Android connectivity", "OPEN", callback=hardware.openNetworkSettings)]
    elif panel == "device":
      items = [text_item("Vehicle transmission", "DISABLED"),
               button_item("Android application settings", "OPEN", callback=hardware.openApplicationSettings)]
    else:
      items = [text_item("Updates", "Install a verified APK through Android"),
               text_item("Vehicle qualification", "Incomplete; engagement blocked")]
    self.scroller = self._child(Scroller(items))

  def _render(self, rect):
    self.scroller.render(rect)
