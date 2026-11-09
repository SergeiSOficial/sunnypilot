#!/usr/bin/env python3
"""Run a reviewed local validation inventory; preserve completed evidence."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
inventory = Path(sys.argv[1]).resolve()
rows = json.loads(inventory.read_text())
for row in rows:
  if row.get("status") != "pending":
    continue
  print("Starting " + row["name"], flush=True)
  log = inventory.parent / (inventory.stem.removesuffix("-validation-driver") + "-" + row["name"] + "-command.log")
  with log.open("w") as stream:
    result = subprocess.run(row["command"], cwd=root, stdout=stream, stderr=subprocess.STDOUT)
  row.update(status="passed" if result.returncode == 0 else "failed", returncode=result.returncode, log=log.name)
  command = row["command"]
  isolation = json.loads(Path(command[command.index("--report") + 1]).read_text())
  row["isolation_status"] = isolation["isolation_status"]
  inventory.write_text(json.dumps(rows, indent=2) + "\n")
  print(json.dumps({key: row[key] for key in ("name", "status", "isolation_status")}), flush=True)
  if row["isolation_status"] != "passed":
    raise SystemExit("Isolation failed; subsequent experiments not run")
