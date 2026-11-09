"""Re-extraction must invalidate shaders left behind by a rejected patch."""
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_ncnn import fresh_source_member


class SourceCacheTest(unittest.TestCase):
  def test_removed_patch_replaces_source_with_fresh_timestamp(self):
    with tempfile.TemporaryDirectory() as directory:
      destination = Path(directory) / "shader.comp"
      destination.write_text("rejected patch")
      old_object_time = 1_000_000_000
      os.utime(destination, (old_object_time, old_object_time))
      archive_bytes = io.BytesIO()
      with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        member = tarfile.TarInfo("shader.comp")
        member.mtime = 1
        original = b"original shader"
        member.size = len(original)
        archive.addfile(member, io.BytesIO(original))
      archive_bytes.seek(0)
      with tarfile.open(fileobj=archive_bytes) as archive:
        archive.extractall(directory, filter=fresh_source_member)
      self.assertEqual(destination.read_bytes(), original)
      self.assertGreater(destination.stat().st_mtime, old_object_time)

  def test_filter_preserves_archive_path_safety(self):
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaises(tarfile.OutsideDestinationError):
        fresh_source_member(tarfile.TarInfo("../escape"), directory)
