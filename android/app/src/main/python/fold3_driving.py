"""Android entry points for the pinned driving processes. No CAN writes."""
import importlib
import time

MODULES = {
  "locationd": "locationd.locationd", "calibrationd": "locationd.calibrationd", "paramsd": "locationd.paramsd",
  "torqued": "locationd.torqued", "lagd": "locationd.lagd", "radard": "controls.radard",
  "plannerd": "controls.plannerd", "selfdrived": "selfdrived.selfdrived",
}


def wait_params(report):
  from openpilot.common.params import Params
  params = Params()
  while params.get("CarParams") is None or params.get("CarParamsSP") is None:
    report.update("waiting_for_car_params")
    time.sleep(.5)


def upstream(name, report):
  """Observe real loop progress in this isolated interpreter, without a timer heartbeat."""
  import openpilot.cereal.messaging as messaging
  wait_params(report)
  report.update("initializing_upstream", upstream_module=MODULES[name])
  original = messaging.SubMaster

  class ObservedSubMaster(original):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.last_return = None

    def update(self, timeout=1000):
      entered = time.monotonic_ns()
      computation = None if self.last_return is None else entered - self.last_return
      result = super().update(timeout)
      returned = time.monotonic_ns()
      if computation is not None:
        report.samples.append(computation)
        report.data["work_samples"] += 1
      report.update("running" if any(self.updated.values()) else "waiting_for_inputs",
        updated_services=[s for s in self.updated if self.updated[s]],
        invalid_services=[s for s in self.valid if not self.valid[s]],
        input_handling_ms=(returned - entered) / 1e6,
        computation_ms=None if computation is None else computation / 1e6,
        complete_cycle_ms=None if self.last_return is None else (returned - self.last_return) / 1e6,
        timing_includes_report_overhead=True)
      self.last_return = returned
      return result

  messaging.SubMaster = ObservedSubMaster
  try:
    importlib.import_module("openpilot.selfdrive." + MODULES[name]).main()
  finally:
    messaging.SubMaster = original


def vehicle(report):
  """Parse the user's unqualified vehicle profile using passive CAN traffic only.

  The pinned Hyundai has firmware fingerprints only. Passive CAN cannot prove
  that identity, so noOutput and the explicit firmware/identity gates remain.
  """
  import openpilot.cereal.messaging as messaging
  from opendbc.car import gen_empty_fingerprint
  from opendbc.car.car_helpers import interfaces
  from opendbc.car.hyundai.values import CAR
  from openpilot.selfdrive.car.card import Car, can_comm_callbacks
  socket = messaging.sub_sock("can", timeout=20)
  received, _ = can_comm_callbacks(socket, None)

  fingerprint, first_packet = gen_empty_fingerprint(), None
  while first_packet is None or time.monotonic() - first_packet < 2:
    packets = received(True)
    for packet in packets:
      for msg in packet:
        if msg.src < 128:
          fingerprint.setdefault(msg.src, {})[msg.address] = len(msg.dat)
          if first_packet is None:
            first_packet = time.monotonic()
    report.update("collecting_passive_can" if first_packet is not None else "waiting_for_passive_can",
      active_queries_enabled=False, firmware_qualified=False, physical_engagement_allowed=False,
      passive_fingerprint_verified=False, identity_basis="user-configured 2019 Ioniq PHEV; firmware-only upstream fingerprint",
      observed_addresses={str(bus): len(addresses) for bus, addresses in fingerprint.items()})

  if first_packet is not None:
    candidate = CAR.HYUNDAI_IONIQ_PHEV_2019
    interface = interfaces[candidate]
    cp = interface.get_params(candidate, fingerprint, [], False, False, False)
    cp_sp = interface.get_params_sp(cp, candidate, fingerprint, [], False, False, False)
    if cp.openpilotLongitudinalControl or not cp.pcmCruise:
      raise PermissionError("Passive Hyundai configuration must preserve factory ACC")
    ci = interface(cp, cp_sp)
    radar = interface.RadarInterface(cp, cp_sp)
    process = Car(ci, radar, receive_only=True)
    report.update("running", passive_can_candidate=str(candidate), factory_acc=True,
      safety_model="noOutput", firmware_qualified=False, calibration_qualified=False)
    def progress(valid, work_ns):
      report.samples.append(work_ns)
      report.data["work_samples"] += 1
      report.update("running" if valid else "waiting_for_valid_can", can_valid=valid, receive_only=True)
    process.card_thread(progress=progress)


def native_check():
  """Execute target NumPy, both EKFs and MPC; imports alone cannot validate an ABI."""
  from pathlib import Path
  import numpy as np
  from openpilot.selfdrive.locationd.models.pose_kf import PoseKalman
  from openpilot.selfdrive.locationd.models.car_kf import CarKalman
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
  from openpilot.cereal import log
  if np.__version__ != "2.5.3":
    raise ValueError(f"NumPy differs from host reference: {np.__version__}")
  np.testing.assert_allclose(np.linalg.solve(np.array([[3., 1.], [1., 2.]]), [9., 8.]), [2., 3.], rtol=1e-12)
  fft_input = np.arange(32, dtype=np.float64)
  np.testing.assert_allclose(np.fft.irfft(np.fft.rfft(fft_input), n=len(fft_input)), fft_input, rtol=1e-12, atol=1e-12)
  generated = str(Path(__file__).parent / "openpilot/selfdrive/locationd/models/generated")
  pose, car = PoseKalman(generated, .8), CarKalman(generated)
  for kalman in (pose, car):
    if not np.isfinite(kalman.filter.state()).all() or not np.isfinite(kalman.filter.covs()).all():
      raise ValueError("Non-finite initial Kalman state")
  mpc = LongitudinalMpc()
  mpc.set_cur_state(15., 0.)
  radar = log.RadarState.new_message()
  mpc.update(radar)
  if mpc.solution_status != 0 or not np.isfinite(mpc.v_solution).all():
    raise ValueError(f"MPC failed: {mpc.solution_status}")
  return {"status": "passed", "numpy": np.__version__, "kalman_models": ["pose", "car"],
          "fft_roundtrip": True, "mpc_status": int(mpc.solution_status), "vehicle_tx_enabled": False}
