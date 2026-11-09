"""Synthetic source fixtures shared by host and debug-only Android replay."""
if __package__:
  from .android_protocol import CHAIN_INBOUND
else:
  from android_protocol import CHAIN_INBOUND


class Sources:
  def __init__(self):
    from opendbc.can import CANPacker
    from opendbc.car import gen_empty_fingerprint, Bus
    from opendbc.car.car_helpers import interfaces
    from opendbc.car.hyundai.values import CAR, DBC
    self.candidate = CAR.HYUNDAI_IONIQ_PHEV_2019
    interface = interfaces[self.candidate]
    cp = interface.get_params(self.candidate, gen_empty_fingerprint(), [], False, False, False)
    cp_sp = interface.get_params_sp(cp, self.candidate, gen_empty_fingerprint(), [], False, False, False)
    self.ci = interface(cp, cp_sp)
    self.ci.update([])  # Register the exact messages read by the pinned parser.
    self.packer = CANPacker(DBC[self.candidate][Bus.pt])
    self.messages = [(state.name, parser.bus) for parser in self.ci.can_parsers.values() if parser
                     for state in parser.message_states.values()]

  def can(self, cruise=True, aeb=False):
    values = {"CGW1": {"CF_Gway_DrvSeatBeltSw": 1}, "SCC11": {"MainMode_ACC": 1, "VSetDis": 72},
              "SCC12": {"ACCMode": int(cruise), "CF_VSM_Warn": int(aeb), "AEB_CmdAct": int(aeb),
                        "CF_VSM_DecCmdAct": int(aeb)}, "ELECT_GEAR": {"Elect_Gear_Shifter": 5},
              "WHL_SPD11": {"WHL_SPD_" + corner: 72 for corner in ("FL", "FR", "RL", "RR")},
              "CLU15": {"CF_Clu_VehicleSpeed": 72}}
    return [self.packer.make_can_msg(name, bus, values.get(name, {})) for name, bus in self.messages]

  def events(self, frame, cruise=True):
    from openpilot.cereal import messaging
    from openpilot.selfdrive.car.card import can_list_to_can_capnp
    events = {"can": messaging.log_from_bytes(can_list_to_can_capnp(self.can(cruise))).as_builder()}
    for name, union, vector in (("accelerometer", "acceleration", [0., 0., -9.81]),
                                 ("gyroscope", "gyroUncalibrated", [0., 0., 0.])):
      event = messaging.new_message(name, valid=True)
      sensor = getattr(event, name)
      sensor.timestamp, sensor.source = event.logMonoTime, "android"
      sensor.init(union).v = vector
      events[name] = event
    if frame % 5 == 0:
      for name in ("modelV2", "driverStateV2", "cameraOdometry"):
        events[name] = messaging.new_message(name, valid=True)
        if name != "driverStateV2":
          getattr(events[name], name).timestampEof = events[name].logMonoTime
      model = events["modelV2"].modelV2
      for field in ("position", "velocity", "acceleration", "orientation", "orientationRate"):
        value = getattr(model, field)
        value.x, value.y, value.z = [20. if field == "velocity" else 0.] * 33, [0.] * 33, [0.] * 33
      from openpilot.selfdrive.modeld.constants import ModelConstants
      predictions = model.meta.disengagePredictions
      predictions.t = ModelConstants.META_T_IDXS
      for field in ("brakeDisengageProbs", "gasDisengageProbs", "steerOverrideProbs",
                    "brake3MetersPerSecondSquaredProbs", "brake4MetersPerSecondSquaredProbs",
                    "brake5MetersPerSecondSquaredProbs", "gasPressProbs", "brakePressProbs"):
        setattr(predictions, field, [0.] * len(ModelConstants.META_T_IDXS))
      odometry = events["cameraOdometry"].cameraOdometry
      odometry.trans, odometry.rot = [20., 0., 0.], [0.] * 3
      odometry.transStd, odometry.rotStd = [.1] * 3, [.01] * 3
      odometry.wideFromDeviceEuler, odometry.wideFromDeviceEulerStd = [0.] * 3, [.01] * 3
      odometry.roadTransformTrans, odometry.roadTransformTransStd = [0., 0., 1.2], [.1] * 3
      for side in ("leftDriverData", "rightDriverData"):
        driver = getattr(events["driverStateV2"].driverStateV2, side)
        driver.faceProb, driver.faceOrientation = 1., [0.] * 3
        driver.facePosition = [0.] * 2
        driver.faceOrientationStd, driver.facePositionStd = [.01] * 3, [.01] * 2
        driver.leftEyeProb = driver.rightEyeProb = 1.
    assert events.keys() <= CHAIN_INBOUND
    return events



def observe(report, phase, name, state):
  samples = report.setdefault("output_samples", {})
  samples[name] = samples.get(name, 0) + 1
  if name == "carState":
    for suffix, value in (("can_valid", state.canValid), ("can_invalid", not state.canValid),
                          ("acc_disabled", not state.cruiseState.enabled), ("acc_enabled", state.cruiseState.enabled)):
      key = phase + "_" + suffix + "_samples"
      report[key] = report.get(key, 0) + int(value)
  if name == "carControl":
    report["unsafe_outputs"] = report.get("unsafe_outputs", 0) + int(state.enabled or state.latActive or state.longActive)
