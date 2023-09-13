"""
Testing for x_epi command line utility
"""

# Load libs
import argparse
import unittest
from x_epi.bin.x_epi_cmd import range_wrapper


class TestRangeWrapper(unittest.TestCase):
    def test_range_wrapper_outside(self):
        range_check = range_wrapper(1, 10)
        with self.assertRaises(argparse.ArgumentTypeError):
            range_check(str(-1))
        with self.assertRaises(argparse.ArgumentTypeError):
            range_check(str(11))

    def test_range_wrapper_inside(self):
        range_check = range_wrapper(1, 10)
        self.assertEqual(range_check(str(5)), 5, "Should be 5")


if __name__ == "__main__":
    unittest.main()
