"""Exercise patched native reader registration and wakeups with OS recorders."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class MsgqWakeupTest(unittest.TestCase):
  def compile_check(self, body, android=False):
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory)
      (work / "msgq").mkdir()
      for filename in ("msgq.cc", "msgq.h"):
        source = subprocess.check_output(["git", "-C", str(ROOT / "msgq_repo"), "show",
          f"e7396e76dadbb49e374d4b664ff6bbb43a39bcb0:msgq/{filename}"])
        (work / "msgq" / filename).write_bytes(source)
      subprocess.run(["git", "apply", "--include=msgq/msgq.cc", str(ROOT / "android/patches/msgq-android.patch")],
                     cwd=work, check=True, capture_output=True)
      if android:
        (work / "msgq/android_ipc.h").write_text(
          '#pragma once\ninline std::string msgq_android_path(const std::string &name) { return name; }\n')
      # Compile the actual pinned implementation, replacing only OS calls with
      # recorders. Standard headers are loaded before changing platform macros.
      headers = "\n".join(line for line in (work / "msgq/msgq.cc").read_text().splitlines()
                          if line.startswith("#include <"))
      (work / "check.cc").write_text(headers + body)
      compiled = subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-I", str(work), str(work / "check.cc"), "-o", str(work / "check")],
                                text=True, capture_output=True)
      self.assertEqual(compiled.returncode, 0, compiled.stderr)
      subprocess.run([str(work / "check")], check=True, capture_output=True)

  @unittest.skipUnless(shutil.which(os.environ.get("CXX", "c++")), "native C++ compiler required")
  def test_empty_registration_slot_cannot_signal_a_process_group(self):
    self.compile_check('''
#include <csignal>
#include <sys/types.h>
#include <sys/syscall.h>
#include <cassert>
static int calls = 0;
static pid_t recipient = -1;
static int recorded_kill(pid_t pid, int signal) {
  assert(signal == SIGUSR2);
  ++calls;
  recipient = pid;
  return 0;
}
#undef __APPLE__
#undef SYS_tkill
#ifndef SYS_gettid
#define SYS_gettid 0
#endif
#define kill recorded_kill
#include "msgq/msgq.cc"
int main() {
  thread_signal(0);
  assert(calls == 0);
  thread_signal(123);
  assert(calls == 1 && recipient == 123);
}
''')

  @unittest.skipUnless(shutil.which(os.environ.get("CXX", "c++")), "native C++ compiler required")
  def test_android_wakeup_retains_reader_process_and_thread(self):
    self.compile_check('''
#include <cstdarg>
#undef __APPLE__
#define __ANDROID__ 1
#ifndef SYS_gettid
#define SYS_gettid 178
#endif
#ifndef SYS_tgkill
#define SYS_tgkill 131
#endif
static int calls = 0;
static unsigned int target_pid = 0, target_tid = 0;
static long recorded_syscall(long number, ...) {
  if (number == SYS_gettid) return 999;
  assert(number == SYS_tgkill);
  va_list args;
  va_start(args, number);
  target_pid = va_arg(args, unsigned int);
  target_tid = va_arg(args, unsigned int);
  assert(va_arg(args, int) == SIGUSR2);
  va_end(args);
  ++calls;
  return 0;
}
#define syscall recorded_syscall
#include "msgq/msgq.cc"
int main() {
  thread_signal(0);
  thread_signal(999); // No registered process, even if the tid exists.
  thread_signal(static_cast<uint64_t>(123) << 32); // No thread.
  assert(calls == 0);
  thread_signal((static_cast<uint64_t>(123) << 32) | 999);
  assert(calls == 1 && target_pid == 123 && target_tid == 999);

  // Exercise actual subscriber registration, not a hand-built reader uid.
  msgq_header_t header{};
  msgq_queue_t queue{};
  queue.num_readers = reinterpret_cast<std::atomic<uint64_t>*>(&header.num_readers);
  queue.write_pointer = reinterpret_cast<std::atomic<uint64_t>*>(&header.write_pointer);
  for (size_t i = 0; i < NUM_READERS; ++i) {
    queue.read_uids[i] = reinterpret_cast<std::atomic<uint64_t>*>(&header.read_uids[i]);
    queue.read_pointers[i] = reinterpret_cast<std::atomic<uint64_t>*>(&header.read_pointers[i]);
    queue.read_valids[i] = reinterpret_cast<std::atomic<uint64_t>*>(&header.read_valids[i]);
  }
  msgq_init_subscriber(&queue);
  uint64_t uid = *queue.read_uids[queue.reader_id];
  assert((uid >> 32) == static_cast<uint32_t>(getpid()));
  assert((uid & 0xFFFFFFFF) == 999);
  thread_signal(uid);
  assert(calls == 2 && target_pid == static_cast<uint32_t>(getpid()) && target_tid == 999);
}
''', android=True)
