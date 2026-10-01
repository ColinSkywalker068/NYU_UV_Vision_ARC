"""Freshness checks must not report a stale/present-only link as healthy."""
import math
from types import SimpleNamespace
import unittest

from check_typec_link import fresh_attitude


class TypeCLinkTest(unittest.TestCase):
    def sample(self, **changes):
        values = dict(valid=True, receive_stamp_ns=1_000_000_000,
                      yaw=0.1, pitch=-0.2, roll=0.0, yaw_rate=0.0, pitch_rate=0.0)
        values.update(changes)
        return SimpleNamespace(**values)

    def test_fresh_valid_sample(self):
        self.assertTrue(fresh_attitude(self.sample(), 1_010_000_000, 990_000_000))

    def test_stale_future_and_missing_stamps(self):
        self.assertFalse(fresh_attitude(self.sample(), 1_600_000_000, 0))
        self.assertFalse(fresh_attitude(self.sample(), 999_999_999, 0))
        self.assertFalse(fresh_attitude(self.sample(receive_stamp_ns=0), 0, 0))

    def test_duplicate_and_out_of_order_samples(self):
        self.assertFalse(fresh_attitude(self.sample(), 1_010_000_000, 1_000_000_000))
        self.assertFalse(fresh_attitude(self.sample(), 1_010_000_000, 1_005_000_000))

    def test_invalid_and_nonfinite_attitude(self):
        self.assertFalse(fresh_attitude(self.sample(valid=False), 1_010_000_000, 0))
        for key in ("yaw", "pitch", "roll", "yaw_rate", "pitch_rate"):
            for value in (math.nan, math.inf):
                self.assertFalse(fresh_attitude(self.sample(**{key: value}), 1_010_000_000, 0))


if __name__ == "__main__":
    unittest.main()
