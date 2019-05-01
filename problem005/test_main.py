import unittest

from .main import remove_duplicate


class TestTemplate(unittest.TestCase):
    def test_example(self) -> None:
        test_case = [
            {"array": [1, 1, 2, 2, 3, 4, 5], "expected": [1, 2, 3, 4, 5]},
            {"array": [1, 1, 1, 1, 1, 1, 1], "expected": [1]},
            {"array": [1, 2, 3, 4, 5, 6, 7], "expected": [1, 2, 3, 4, 5, 6, 7]},
            {"array": [1, 2, 1, 1, 1, 1, 1], "expected": [1, 2]},
        ]
        for params in test_case:
            with self.subTest(params=params):
                self.assertListEqual(
                    params["expected"], remove_duplicate(params["array"])
                )
