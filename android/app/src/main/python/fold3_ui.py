"""Run the existing Raylib UI on NativeActivity's rendering thread."""
import json
import faulthandler
import math
from collections import deque
import os
from pathlib import Path
import time
import traceback


def main(data_dir, simulation=False, preview=False):
  from fold3_runtime import configure
  configure(data_dir, simulation)
  os.environ["FPS"] = "30"
  os.environ["BIG"] = "1"
  path = Path(data_dir) / "ui-report.json"
  def save_report():
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)
  report = {"pid": os.getpid(), "backend": "Raylib Android NativeActivity", "vehicle_tx_enabled": False}
  camera_views = []
  try:
    from openpilot.selfdrive.ui.ui import main as upstream_main
  except Exception:
    report.update(status="blocked", full_upstream_ui=False, error=traceback.format_exc())
  else:
    report.update(status="starting", full_upstream_ui=True)
    save_report()
    frames, started = 0, time.monotonic()
    intervals, cpu_samples = deque(maxlen=3000), deque(maxlen=3000)
    startup_stack = (Path(data_dir) / "ui-startup-stack.txt").open("w")
    faulthandler.dump_traceback_later(30, file=startup_stack)
    def progress(frame_time, cpu_time):
      nonlocal frames
      import pyray as rl
      from java import jclass
      if preview and not simulation:
        if not camera_views:
          from openpilot.selfdrive.ui.onroad.cameraview import CameraView
          from openpilot.cereal.visionipc import VisionStreamType
          class FreshCameraView(CameraView):
            def _render_textures(self, source, destination):
              if 0 <= time.monotonic_ns() - self.client.timestamp_sof <= 100_000_000:
                super()._render_textures(source, destination)
          camera_views.extend([FreshCameraView("camerad", stream) for stream in
                               (VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_CABIN)])
        size = rl.get_screen_width() / 2
        for i, view in enumerate(camera_views):
          rect = rl.Rectangle(i * size, 84, size, rl.get_screen_height() - 150)
          rl.draw_rectangle_rec(rect, rl.BLACK)
          view.render(rect)
        report["camera_preview"] = [{"stream": name, "frame_id": view.client.frame_id,
                                     "age_ms": (time.monotonic_ns() - view.client.timestamp_sof) / 1e6,
                                     "fresh": 0 <= time.monotonic_ns() - view.client.timestamp_sof <= 100_000_000}
                                    for name, view in zip(("road", "driver"), camera_views, strict=True)]
        rl.draw_text_ex(rl.get_font_default(), "UNCALIBRATED CAMERA DIAGNOSTICS - DRIVER AND ROAD MODELS UNQUALIFIED",
                        rl.Vector2(12, 48), 22, 1, rl.YELLOW)
      rl.draw_rectangle(0, 0, rl.get_screen_width(), 42, rl.Color(120, 0, 0, 240))
      rl.draw_text_ex(rl.get_font_default(), "SIMULATION - VEHICLE TRANSMISSION DISABLED" if simulation else "VEHICLE TRANSMISSION DISABLED",
                      rl.Vector2(12, 8), 26, 1, rl.WHITE)
      frames += 1
      intervals.append(frame_time)
      cpu_samples.append(cpu_time)
      if frames == 1:
        faulthandler.cancel_dump_traceback_later()
      if frames == 1 or frames % 30 == 0:
        index = math.ceil(len(intervals) * .99) - 1
        report.update(status="running", frames=frames, mean_fps=frames / (time.monotonic() - started),
                      width=rl.get_screen_width(), height=rl.get_screen_height(),
                      surface_size=list(jclass("ai.sunnypilot.fold3.AndroidHardware").surfaceSize()),
                      last_frame_seconds=frame_time, last_cpu_seconds=cpu_time,
                      recent_p99_frame_ms=sorted(intervals)[index] * 1000,
                      recent_p99_cpu_ms=sorted(cpu_samples)[index] * 1000)
        save_report()
    try:
      upstream_main(progress=progress)
    except Exception:
      report.update(status="failed", error=traceback.format_exc())
      save_report()
      raise
    finally:
      for view in camera_views:
        view.close()
      faulthandler.cancel_dump_traceback_later()
      startup_stack.close()
    return
  # Retain a useful, honest diagnostic surface when an upstream UI dependency
  # fails. This is the same GUI application and Android backend, not a driving UI.
  save_report()
  import pyray as rl
  from openpilot.system.ui.lib.application import gui_app
  gui_app.init_window("sunnypilot Android runtime", fps=30)
  frames = 0
  start = time.monotonic()
  for should_render, frame_time, cpu_time in gui_app.render():
    if should_render:
      rl.draw_text_ex(rl.get_font_default(), "Vehicle transmission disabled", rl.Vector2(40, 50), 40, 1, rl.WHITE)
      rl.draw_text_ex(rl.get_font_default(), "Driving UI blocked: see ui-report.json", rl.Vector2(40, 110), 28, 1, rl.YELLOW)
      manager = Path(data_dir) / "manager.json"
      if manager.exists():
        rl.draw_text_ex(rl.get_font_default(), manager.read_text()[:1200], rl.Vector2(40, 170), 18, 1, rl.WHITE)
      frames += 1
      if frames % 30 == 0:
        report.update(frames=frames, mean_fps=frames / (time.monotonic() - start),
                      width=rl.get_screen_width(), height=rl.get_screen_height(),
                      last_frame_seconds=frame_time, last_cpu_seconds=cpu_time)
        save_report()
