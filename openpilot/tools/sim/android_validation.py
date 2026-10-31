#!/usr/bin/env python3
"""Exercise Android's real controlsd with isolated synthetic Cereal inputs.

--metadrive closes steering feedback through the existing MetaDrive world.
Perception, calibration and driver images are fixtures: this never qualifies
the visual driving stack, Hyundai CAN integration or a physical vehicle.
"""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

from openpilot.tools.sim.lib.android_protocol import receive, send
from openpilot.tools.sim.lib.android_protocol import OUTBOUND
from openpilot.tools.sim.android_bridge import connect


def inputs(elapsed, speed, steering, min_steer_speed, desired_curvature=None):
  import openpilot.cereal.messaging as messaging
  speed, steering = float(speed), float(steering)
  messages = {}
  for name in ("carState", "modelV2", "selfdriveState", "selfdriveStateSP", "vehicleParameters", "lateralDelay",
               "extrinsicsCalibration", "driverStateV2", "longitudinalPlan", "carOutput", "radarState"):
    messages[name] = messaging.new_message(name, valid=True)
  cs = messages["carState"].carState
  cs.canValid, cs.vEgo, cs.steeringAngleDeg = True, speed, steering
  cs.gearShifter, cs.vCruise, cs.vCruiseCluster = "drive", 72., 72.
  cs.cruiseState.enabled, cs.cruiseState.available = True, True
  model = messages["modelV2"].modelV2
  model.action.desiredCurvature = .0005 * math.cos(elapsed * .5) if desired_curvature is None else desired_curvature
  model.velocity.x, model.acceleration.y = [speed] * 33, [0.] * 33
  messages["selfdriveState"].selfdriveState.active = speed >= min_steer_speed
  messages["selfdriveState"].selfdriveState.enabled = True
  lp = messages["vehicleParameters"].vehicleParameters
  lp.valid, lp.stiffnessFactor, lp.steerRatio = True, 1., 13.73
  messages["lateralDelay"].lateralDelay.lateralDelay = .2
  calib = messages["extrinsicsCalibration"].extrinsicsCalibration
  calib.calStatus, calib.rpyCalib = "calibrated", [0., 0., 0.]
  calib.wideFromDeviceEuler = [0., 0., 0.]
  for side in ("leftDriverData", "rightDriverData"):
    driver = getattr(messages["driverStateV2"].driverStateV2, side)
    driver.faceProb, driver.faceOrientation = 1., [0., 0., 0.]
  events = messaging.new_message("onroadEvents", 0, valid=True)
  messages["onroadEvents"] = events
  return {name: base64.b64encode(event.to_bytes()).decode() for name, event in messages.items()}


def validate(serial, adb, duration, fault, metadrive, report_path, port=0):
  # Set these before importing Params or any simulator module. Never touch the
  # host's ordinary Params, controls, CAN, Panda or VisionIPC namespace.
  root = tempfile.TemporaryDirectory(prefix="fold3_sim_validation_")
  os.environ["OPENPILOT_PREFIX"] = Path(root.name).name
  os.environ["PARAMS_ROOT"] = root.name + "/params"
  from openpilot.cereal import messaging
  from openpilot.common.params import Params
  from openpilot.tools.sim.lib.common import SimulatorState
  report = {"status": "starting", "vehicle_tx_enabled": False, "physical_vehicle_verified": False,
            "perception_verified": False, "synthetic_driver_and_calibration": True, "factory_acc_fixture": True,
            "metadrive": metadrive, "fault": fault, "packets": 0, "active_samples": 0,
            "inactive_after_fault": 0, "longitudinal_active_samples": 0, "max_abs_steering_deg": 0.,
            "below_minimum_active_samples": 0,
            "max_abs_torque": 0., "steering_plant": "synthetic damped torque-to-angle; not a Hyundai vehicle model"}
  world, bridge = None, None
  forwarded = subprocess.run([adb, "-s", serial, "forward", "--no-rebind", f"tcp:{port}", "localabstract:sunnypilot_fold3_sim"],
                             check=True, capture_output=True, text=True)
  if port == 0:
    port = int(forwarded.stdout.strip())
  report["adb_port"] = port
  try:
    state = SimulatorState()
    if metadrive:
      from multiprocessing import Queue
      from openpilot.tools.sim.bridge.metadrive.metadrive_bridge import MetaDriveBridge, MapGenerateMethod, straight_block
      bridge = MetaDriveBridge(False, False)
      Params().put_bool("AlphaLongitudinalEnabled", False, block=True)
      # A straight validation segment lets the bounded oscillating curvature
      # fixture exercise steering without pretending to perceive a curved map.
      map_config = {"type": MapGenerateMethod.PG_MAP_FILE, "lane_num": 2, "lane_width": 4.5,
                    "config": [None] + [straight_block(60.) for _ in range(math.ceil(max(500., duration * 25.) / 60.))]}
      world = bridge.spawn_world(Queue(), map_config=map_config, initial_speed=20.)
      origin = tuple(world.vehicle_position)
      report["metadrive_world_started"] = True
      report["perception_source"] = "synthetic straight-lane curvature from MetaDrive ground truth; not camera inference"
    start, sequence, steering = time.monotonic(), 0, 0.
    connection, profile = connect(port)
    with connection:
      report["android_profile"] = profile
      min_steer_speed = profile["min_steer_speed_m_s"]
      while (elapsed := time.monotonic() - start) < duration:
        iteration = time.monotonic()
        if world:
          world.read_state()
          world.read_sensors(state)
          speed = state.speed
        else:
          speed = 20.
        active_fault = fault if elapsed > duration * .65 else None
        measured_steering = state.steering_angle * 8 if world else steering
        desired_curvature = None
        if world and state.valid:
          # The straight map runs along +X. Openpilot's desired curvature has
          # the opposite sign to MetaDrive's positive steering convention.
          # Feed a bounded ground-truth lane fixture, with measured position
          # and heading feedback, instead of a drifting open-loop sine input.
          lateral = float(world.vehicle_position[1] - origin[1])
          heading = math.remainder(math.radians(state.bearing), 2 * math.pi)
          target = .2 * math.sin(elapsed * .5)
          lookahead = max(20., speed * 1.5)
          desired_curvature = max(-.005, min(.005, 2 * (lateral - target + lookahead * heading) / lookahead**2))
          report["max_lateral_error_m"] = max(report.get("max_lateral_error_m", 0.), abs(lateral))
        send(connection, {"sequence": sequence,
                          "messages": inputs(elapsed, speed, measured_steering, min_steer_speed, desired_curvature), "fault": active_fault})
        reply = receive(connection)
        if reply.get("sequence") != sequence or reply.get("vehicle_tx_enabled") is not False:
          raise ValueError("invalid simulator response")
        outputs = reply.get("messages", {})
        if not isinstance(outputs, dict) or not outputs.keys() <= set(OUTBOUND):
          raise ValueError("Android returned disallowed simulator services")
        for service, payload in outputs.items():
          event = messaging.log_from_bytes(base64.b64decode(payload, validate=True))
          if event.which() != service:
            raise ValueError("Cereal service mismatch")
          if service == "carControl":
            cc = event.carControl
            report["active_samples"] += int(cc.latActive and not active_fault)
            report["inactive_after_fault"] += int(not cc.latActive and bool(active_fault))
            report["longitudinal_active_samples"] += int(cc.longActive)
            report["below_minimum_active_samples"] += int(cc.latActive and speed < min_steer_speed)
            if cc.latActive and speed < min_steer_speed:
              raise AssertionError("lateral control below Hyundai minimum steering speed")
            if cc.longActive:
              raise AssertionError("phone attempted longitudinal control in factory ACC simulation")
            # Hyundai's torque controller intentionally returns no angle. Use
            # an explicit synthetic steering plant, feeding back the world's
            # measured angle. This exercises torque commands, not an EPS model.
            torque = float(cc.actuators.torque) if cc.latActive else 0.
            if not math.isfinite(torque) or abs(torque) > 1.:
              raise AssertionError("invalid normalized steering torque")
            steering = max(-120., min(120., measured_steering + (.01 * (120. * torque - .7 * measured_steering))))
            report["max_abs_torque"] = max(report["max_abs_torque"], abs(torque))
            if not math.isfinite(steering):
              raise AssertionError("non-finite steering")
            report["max_abs_steering_deg"] = max(report["max_abs_steering_deg"], abs(steering))
        if world:
          state.is_engaged = report["active_samples"] > 0 and active_fault is None
          # Factory ACC is an independent simulator speed controller. No phone
          # acceleration or brake command reaches the world.
          acceleration = max(-1., min(1., (20. - speed) * .25)) if active_fault is None else -.3
          world.apply_controls(steering, max(0., acceleration), max(0., -acceleration))
          report["max_speed_m_s"] = max(report.get("max_speed_m_s", 0.), speed)
          if world.exit_event.is_set():
            raise RuntimeError("MetaDrive world stopped unexpectedly")
        sequence += 1
        report["packets"] = sequence
        time.sleep(max(0., .01 - (time.monotonic() - iteration)))
    report["status"] = "passed" if (report["active_samples"] > 10 and report["longitudinal_active_samples"] == 0
                                            and report["max_abs_torque"] > 1e-5 and report["max_abs_steering_deg"] > .01
                                            and (not fault or report["inactive_after_fault"] > 10)) else "failed"
    report["duration_s"] = time.monotonic() - start
    report["qualification_blocker"] = "Synthetic perception/DM/calibration fixture; no visual closed-loop or physical vehicle qualification"
  except Exception as exc:
    report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    raise
  finally:
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    if world:
      world.close("Android controller fixture finished")
    subprocess.run([adb, "-s", serial, "forward", "--remove", f"tcp:{port}"], check=False, capture_output=True)
    root.cleanup()
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--duration", type=float, default=20)
  parser.add_argument("--port", type=int, default=0, help="0 asks ADB for a free host port")
  parser.add_argument("--fault", choices=("cancel", "stale"))
  parser.add_argument("--metadrive", action="store_true")
  parser.add_argument("--report", default="android-controller-validation.json")
  args = parser.parse_args()
  if not 5 <= args.duration <= 3600:
    parser.error("duration must be 5–3600 seconds")
  if not 0 <= args.port <= 65535:
    parser.error("port must be 0–65535")
  print(json.dumps(validate(args.serial, args.adb, args.duration, args.fault, args.metadrive, args.report, args.port), indent=2))


if __name__ == "__main__":
  main()
