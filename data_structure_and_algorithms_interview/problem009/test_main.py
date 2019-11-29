import unittest

from .main import reverse_list


class TestReverse(unittest.TestCase):
    def test_example(self) -> None:
        test_case = [
            {"array": [1, 2, 3, 4, 5], "expected": [5, 4, 3, 2, 1]},
            {"array": [1, 2, 3, 4], "expected": [4, 3, 2, 1]},
        ]
        for params in test_case:
            with self.subTest(params=params):
                self.assertListEqual(params["expected"], reverse_list(params["array"]))
