import sys
import unittest

from .main import search_min_max


class TestSearchMinMax(unittest.TestCase):
    """ test case of search min and max value.
    """
    def test_example(self) -> None:
        test_case = [
            {"array": [-20, 34, 21, -87, 92, sys.maxsize], "expected": (-87, sys.maxsize)},
            {"array": [10, -sys.maxsize, -2], "expected": (-sys.maxsize, 10)},
            {"array": [sys.maxsize, 40, sys.maxsize], "expected": (40, sys.maxsize)},
            {"array": [1, -1, 0], "expected": (-1, 1)},
        ]
        for params in test_case:
            with self.subTest(params=params):
                self.assertEqual(params["expected"], search_min_max(params["array"]))


if __name__ == "__main__":
    unittest.main()
