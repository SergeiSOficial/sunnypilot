#!/usr/bin/env python3
"""Extract the pinned, read-only Panda wire decoder without importing USB/SPI drivers."""
import ast
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


class KeepChecksumChecks(ast.NodeTransformer):
  def visit_Assert(self, node):
    # Keep the pinned condition/error even when Python uses optimized bytecode.
    return ast.If(test=ast.UnaryOp(op=ast.Not(), operand=node.test),
                  body=[ast.Raise(exc=ast.Call(func=ast.Name(id="AssertionError", ctx=ast.Load()), args=[node.msg], keywords=[]))], orelse=[])


def generate():
  tree = ast.parse((ROOT / "panda/python/__init__.py").read_text())
  selected = []
  for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("CANPACKET_HEAD_SIZE", "DLC_TO_LEN") for t in node.targets):
      selected.append(node)
    if isinstance(node, ast.FunctionDef) and node.name in ("calculate_checksum", "unpack_can_buffer"):
      selected.append(KeepChecksumChecks().visit(node))
  panda = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Panda")
  body = []
  for node in panda.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and (t.id.startswith("HEALTH_FLAG_") or t.id == "REQUEST_IN") for t in node.targets):
      if node.targets[0].id == "REQUEST_IN":
        if ast.unparse(node.value) != "usb1.ENDPOINT_IN | usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE":
          raise ValueError("Pinned USB request type changed")
        node.value = ast.Constant(0xc0)
      body.append(node)
    if isinstance(node, ast.FunctionDef) and node.name == "health":
      node.decorator_list = []  # The caller validates the packet versions before decoding.
      body.append(node)
  health_source = (ROOT / "panda/board/health.h").read_text().split("struct __attribute__((packed)) health_t {", 1)[1].split("};", 1)[0]
  formats = {"uint8_t": "B", "uint16_t": "H", "uint32_t": "I", "float": "f"}
  fields = [re.fullmatch(r"(uint8_t|uint16_t|uint32_t|float)\s+\w+;", line.strip()) for line in health_source.splitlines() if line.strip()]
  if not all(fields):
    raise ValueError("Pinned Panda health structure changed")
  fmt = "<" + "".join(formats[m[1]] for m in fields)
  body.insert(0, ast.parse(f"HEALTH_STRUCT = struct.Struct({fmt!r})").body[0])
  selected.append(ast.ClassDef(name="Panda", bases=[], keywords=[], body=body, decorator_list=[]))
  versions = [int.from_bytes(hashlib.sha256((ROOT / path).read_bytes().replace(b"\r", b"")).digest()[:4], "little")
              for path in ("opendbc_repo/opendbc/safety/can.h", "panda/board/health.h")]
  return ("# Generated from the pinned Panda source by android/scripts/stage_panda.py.\nimport struct\n" +
          f"PACKET_VERSIONS = {tuple(versions)!r}\n" + ast.unparse(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))) + "\n")


if __name__ == "__main__":
  destination = ROOT / "android/build/native/panda_protocol/fold3_panda_protocol.py"
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(generate())
