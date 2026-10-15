"""Offline session collector integration test, using appended synthetic log files."""
import json
import importlib.util
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest


class SessionTests(unittest.TestCase):
    def test_records_until_operator_interrupt(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            logs = repo / 'vision_logs'; logs.mkdir()
            source = logs / 'vision_serial_runtime_test.log'
            source.write_text('old\n')
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('record-tracking-session.py')),
                                        '--repo', str(repo), '--seconds', '0', '--max-mib', '0',
                                        '--video-max-mib', '0', '--preview-port', '1'],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertIn('Recording existing logs', process.stdout.readline())
                with source.open('a') as out:
                    out.write('manual-stop-session\n')
                time.sleep(.6)
                self.assertIsNone(process.poll())
                process.send_signal(signal.SIGINT)
                _, err = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, err)
                session = next((logs / 'sessions').iterdir())
                self.assertEqual((session / 'bridge.log').read_text(), 'manual-stop-session\n')
                manifest = json.loads((session / 'manifest.json').read_text())
                self.assertEqual(manifest['stop_reason'], 'stopped_by_user')
                self.assertTrue(manifest['capture_until_stopped'])
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()

    def test_video_has_no_automatic_limit_in_manual_mode(self):
        spec = importlib.util.spec_from_file_location('session_collector', Path(__file__).with_name('record-tracking-session.py'))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        manual = module.video_command('ffmpeg', 8080, 0, 0, Path('/tmp/overlay.mkv'))
        self.assertNotIn('-t', manual)
        self.assertNotIn('-fs', manual)
        bounded = module.video_command('ffmpeg', 8080, 120, 1024, Path('/tmp/overlay.mkv'))
        self.assertEqual(bounded[bounded.index('-t') + 1], '120')
        self.assertEqual(bounded[bounded.index('-fs') + 1], str(1024 * 1024 * 1024))

    def test_captures_only_new_bytes_across_rotation(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            logs = repo / 'vision_logs'; logs.mkdir()
            source = logs / 'vision_serial_runtime_test.log'
            source.write_text('old\n')
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('record-tracking-session.py')),
                                        '--repo', str(repo), '--seconds', '2', '--preview-port', '1'],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertIn('Recording existing logs', process.stdout.readline())
                with source.open('a') as out:
                    out.write('first\n')
                time.sleep(.4)
                source.rename(source.with_suffix('.log.1'))
                source.write_text('second\n')
                _, err = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, err)
                session = next((logs / 'sessions').iterdir())
                self.assertEqual((session / 'bridge.log').read_text(), 'first\nsecond\n')
                manifest = json.loads((session / 'manifest.json').read_text())
                self.assertEqual(manifest['stop_reason'], 'duration_complete')
                self.assertIn('error', json.loads((session / 'preview.jsonl').read_text().splitlines()[0]))
            finally:
                if process.poll() is None:
                    process.kill(); process.wait()


if __name__ == '__main__':
    unittest.main()
