import unittest

from .main import quick_sort


class TestQuickSort(unittest.TestCase):
    def test_example(self) -> None:
        test_case = [
            {"array": [3, 2, 5, 1, 4], "expected": [1, 2, 3, 4, 5]}
        ]
        for params in test_case:
            with self.subTest(params=params):
                self.assertEqual(params["expected"], quick_sort(params["array"]))
