#!/usr/bin/env python3
"""Exercise startup selection using temporary PTYs, never controller devices."""
import os
import pty
import subprocess
import tempfile
import unittest
from pathlib import Path

HELPER = Path(__file__).with_name("serial-port-selection.sh")


class SerialSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.master, self.slave = pty.openpty()
        self.device = os.ttyname(self.slave)

    def tearDown(self):
        os.close(self.master)
        os.close(self.slave)
        self.temp.cleanup()

    def board(self, serial):
        path = self.directory / f"usb-NYUSH_Vision_Comm_port_{serial}-if00"
        path.symlink_to(self.device)
        return path

    def resolve(self, requested="auto"):
        return subprocess.run(
            ["bash", "-c", 'source "$1"; cv_resolve_serial_port "$2" "$3"',
             "selection-test", str(HELPER), str(requested), str(self.directory)],
            text=True, capture_output=True, check=False)

    def test_unique_and_replacement(self):
        first = self.board("OLD123")
        self.assertEqual(self.resolve().stdout.strip(), str(first))
        first.unlink()
        second = self.board("NEW456")
        self.assertEqual(self.resolve(first).stdout.strip(), str(second))

    def test_missing_waits(self):
        self.assertEqual(self.resolve().returncode, 1)

    def test_ambiguity_refused(self):
        self.board("ONE")
        self.board("TWO")
        result = self.resolve()
        self.assertEqual(result.returncode, 2)
        self.assertIn("Ambiguous", result.stderr)

    def test_explicit_wins(self):
        first = self.board("ONE")
        self.board("TWO")
        self.assertEqual(self.resolve(first).stdout.strip(), str(first))

    def test_unrelated_dead_and_regular_ignored(self):
        (self.directory / "usb-STLINK-if00").symlink_to(self.device)
        (self.directory / "usb-NYUSH_Vision_Comm_port_DEAD-if00").symlink_to("/missing")
        (self.directory / "usb-NYUSH_Vision_Comm_port_FILE-if00").write_text("not a device")
        self.assertEqual(self.resolve().returncode, 1)

    def test_custom_missing_does_not_fallback(self):
        self.board("ONE")
        requested = self.directory / "custom-uart"
        self.assertEqual(self.resolve(requested).stdout.strip(), str(requested))


if __name__ == "__main__":
    unittest.main()
