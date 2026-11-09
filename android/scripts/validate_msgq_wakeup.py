#!/usr/bin/env python3
"""Check the patched msgq wakeup against an Android device's real kernel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PIN = "e7396e76dadbb49e374d4b664ff6bbb43a39bcb0"

CHECK = r'''
#include <poll.h>
#include <sys/wait.h>
#include "msgq/msgq.cc"
static int acknowledgement;
static void received(int) {
  const char byte = 's';
  (void)write(acknowledgement, &byte, 1);
}
int main() {
  int ready[2], done[2];
  assert(pipe(ready) == 0 && pipe(done) == 0);
  pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    alarm(5);
    close(ready[0]);
    close(done[1]);
    acknowledgement = ready[1];
    struct sigaction action{};
    action.sa_handler = received;
    sigemptyset(&action.sa_mask);
    assert(sigaction(SIGUSR2, &action, nullptr) == 0);
    assert(write(ready[1], "r", 1) == 1);
    char byte;
    while (read(done[0], &byte, 1) == -1 && errno == EINTR) {}
    _exit(0);
  }
  alarm(10);
  close(ready[1]);
  close(done[0]);
  char byte;
  assert(read(ready[0], &byte, 1) == 1 && byte == 'r');
  errno = 0;
  thread_signal(0);
  thread_signal(child);  // No registered process.
  thread_signal(static_cast<uint64_t>(child) << 32);  // No thread.
  assert(errno == 0);
  // A live thread in another process must not be woken by a stale registration.
  thread_signal((static_cast<uint64_t>(getpid()) << 32) | child);
  assert(errno == ESRCH);
  struct pollfd event{ready[0], POLLIN, 0};
  assert(poll(&event, 1, 100) == 0);
  thread_signal((static_cast<uint64_t>(child) << 32) | child);
  assert(poll(&event, 1, 1000) == 1);
  assert(read(ready[0], &byte, 1) == 1 && byte == 's');
  assert(write(done[1], "d", 1) == 1);
  int status;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  puts("{\"status\":\"passed\",\"empty_slots_ignored\":true,\"wrong_process_rejected\":true,\"registered_reader_woken\":true}");
}
'''


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--serial", required=True)
  parser.add_argument("--adb", default="adb")
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  sdk = Path(os.environ.get("ANDROID_HOME", Path.home() / "Library/Android/sdk"))
  compiler = sdk / "ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android35-clang++"
  patch = ROOT / "android/patches/msgq-android.patch"
  remote = f"/data/local/tmp/fold3-msgq-wakeup-{os.getpid()}"
  adb = [args.adb, "-s", args.serial]
  result = {"status": "failed", "serial": args.serial, "msgq_pin": PIN,
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}
  try:
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory)
      (work / "msgq").mkdir()
      for filename in ("msgq.cc", "msgq.h"):
        (work / "msgq" / filename).write_bytes(subprocess.check_output(
          ["git", "-C", str(ROOT / "msgq_repo"), "show", f"{PIN}:msgq/{filename}"]))
      subprocess.run(["git", "apply", "--include=msgq/msgq.cc", "--include=msgq/android_ipc.h", str(patch)],
                     cwd=work, check=True, capture_output=True)
      (work / "check.cc").write_text(CHECK)
      binary = work / "check"
      subprocess.run([str(compiler), "-std=c++17", "-O2", "-static-libstdc++", "-I", str(work),
                      str(work / "check.cc"), "-o", str(binary)], check=True, capture_output=True, timeout=60)
      result["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
      subprocess.run([*adb, "push", str(binary), remote], check=True, capture_output=True, timeout=30)
      subprocess.run([*adb, "shell", "chmod", "700", remote], check=True, capture_output=True, timeout=30)
      output = subprocess.check_output([*adb, "shell", remote], text=True, timeout=15)
      result.update(json.loads(output))
  except (OSError, ValueError, subprocess.SubprocessError) as exc:
    result["error"] = str(exc)
    if isinstance(exc, subprocess.CalledProcessError):
      output = exc.stderr or exc.output or b""
      result["details"] = (output.decode(errors="replace") if isinstance(output, bytes) else output)[-4000:]
  finally:
    subprocess.run([*adb, "shell", "rm", "-f", remote], capture_output=True, timeout=30)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
  raise SystemExit(main())
