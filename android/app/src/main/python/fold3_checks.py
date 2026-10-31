"""Small on-device checks, independent of camera access or physical Panda hardware."""
import json
from pathlib import Path


def ort():
  import numpy as np
  from fold3_ort import OrtModel
  import fold3_ort_native
  path = Path(__file__).parent / "fold3_reference/runtime-check.onnx"
  model = OrtModel(path)
  for offset in (0., 17.):
    x = np.arange(4, dtype=np.float32).reshape(1, 4) + offset
    np.testing.assert_array_equal(model({"x": x})["y"], x * 2)
  for bad in ({}, {"x": np.ones((1, 4), dtype=np.uint8)}, {"x": np.zeros((2, 2), dtype=np.float32)},
              {"x": np.zeros((1, 4), dtype=np.float32), "extra": np.zeros(1)}):
    try:
      model(bad)
    except (ValueError, RuntimeError):
      pass
    else:
      raise AssertionError("Invalid ORT input accepted")
  try:
    fold3_ort_native.create(str(path), "cpu", 0)
  except ValueError:
    pass
  else:
    raise AssertionError("Invalid thread count accepted")
  return json.dumps({"status": "passed", "changed_inputs": 2, "rejected_invalid_inputs": 5, "vehicle_tx_enabled": False})


def panda_decoder():
  from fold3_panda import Decoder, require_silent
  from fold3_panda_protocol import Panda, unpack_can_buffer
  assert unpack_can_buffer(bytes.fromhex("201809000020aabb")) == ([(0x123, bytes.fromhex("aabb"), 0)], b"")
  health = Decoder(lambda request, value, length: bytes(Panda.HEALTH_STRUCT.size)).health()
  require_silent(health)
  for key in ("controls_allowed", "controls_allowed_lateral", "controls_allowed_longitudinal", "safety_mode", "faults", "fault_status"):
    try:
      require_silent({**health, key: 1})
    except PermissionError:
      pass
    else:
      raise AssertionError(f"Unsafe {key} accepted")
  return json.dumps({"status": "passed", "unsafe_states_rejected": 6, "physical_hardware_verified": False, "vehicle_tx_enabled": False})


def output_messages(directory=None):
  """Exercise upstream output parsing without pretending fixture images are live."""
  import numpy as np
  from openpilot.cereal import log, messaging
  from openpilot.selfdrive.modeld.dmonitoringmodeld import slice_outputs, parse_model_output, get_driverstate_packet
  from openpilot.selfdrive.modeld.parse_model_outputs import Parser
  from openpilot.selfdrive.modeld.fill_model_msg import PublishState, fill_model_msg, fill_pose_msg
  root = Path(directory) if directory else Path(__file__).parent / "fold3_reference"
  metadata = json.loads((root / "manifest.json").read_text())["models"]
  messages = []
  for name in ("dmonitoring_model", "driving_supercombo"):
    slices = {key: slice(*value) for key, value in metadata[name]["output_slices"].items()}
    raw = np.load(root / f"{name}-outputs.npz")["outputs"].astype(np.float32)
    if name == "dmonitoring_model":
      parsed = parse_model_output(slice_outputs(raw.reshape(-1), slices))
      parsed["raw_pred"] = b""
      message = get_driverstate_packet(parsed, 42, 1000, .01, 0.)
      message.valid = False
      assert len(message.driverStateV2.rightDriverData.faceOrientation) == 3
      messages.append(message)
    else:
      parsed = Parser().parse_outputs({key: raw[:, value] for key, value in slices.items()})
      message, pose = messaging.new_message("modelV2", valid=False), messaging.new_message("cameraOdometry", valid=False)
      fill_model_msg(message, parsed, log.ModelDataV2.Action(), PublishState(), 42, 43, 42, 0., 1000, .01, False)
      fill_pose_msg(pose, parsed, 42, 0, 1000, False)
      assert len(message.modelV2.position.x) == 33
      assert pose.cameraOdometry.frameId == 42
      messages.extend((message, pose))
  for message in messages:
    assert not message.valid
    with log.Event.from_bytes(message.to_bytes()) as received:
      assert not received.valid
  return json.dumps({"status": "passed", "invalid_messages": [message.which() for message in messages], "vehicle_tx_enabled": False})
