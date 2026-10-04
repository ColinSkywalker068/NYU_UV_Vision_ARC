"""Offline session collector integration test, using appended synthetic log files."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


class SessionTests(unittest.TestCase):
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
