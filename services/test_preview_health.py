#!/usr/bin/env python3
"""Offline preview-only tests: no socket, service, ROS, or robot access."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("preview_health", Path(__file__).with_name("preview-health.py"))
health = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health)


class PreviewTests(unittest.TestCase):
    def run_check(self, advancing=True, jpeg=True, missing=False):
        count = 0
        def response(url, **kwargs):
            nonlocal count
            if missing:
                raise OSError("connection refused")
            if url.endswith("stats.json"):
                count += int(advancing)
                return io.BytesIO(json.dumps({"frame_count": count}).encode())
            return io.BytesIO(b"\xff\xd8" if jpeg else b"not JPEG")
        with patch.object(health.urllib.request, "build_opener") as factory:
            factory.return_value.open.side_effect = response
            return health.check("http://127.0.0.1:8080", .6)

    def test_live(self):
        self.assertEqual(self.run_check(), 0)

    def test_frozen(self):
        self.assertEqual(self.run_check(advancing=False), 1)

    def test_bad_image(self):
        self.assertEqual(self.run_check(jpeg=False), 1)

    def test_missing_server(self):
        self.assertEqual(self.run_check(missing=True), 1)

    def test_custom_port(self):
        with patch.dict(health.os.environ, {"CV_STM_EXTRA_LAUNCH_ARGS": "preview_http_port:=8090"}):
            self.assertEqual(health.preview_port(), 8090)


if __name__ == "__main__":
    unittest.main()
