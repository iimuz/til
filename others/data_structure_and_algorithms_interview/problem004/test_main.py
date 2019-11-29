import unittest

from .main import search_pair


class TestSearchPair(unittest.TestCase):
    """ リストから合計値となるペアを探索するテスト
    """

    def test_example(self) -> None:
        test_case = [
            {
                "array": [2, 4, 3, 5, 7, 8, 9],
                "sum_val": 7,
                "expected": [[2, 5], [4, 3]],
            },
            {
                "array": [2, 4, 3, 5, 6, -2, 4, 7, 8, 9],
                "sum_val": 7,
                "expected": [[2, 5], [4, 3], [3, 4], [-2, 9]],
            },
        ]
        for params in test_case:
            with self.subTest(params=params):
                self.assertEqual(
                    params["expected"], search_pair(params["array"], params["sum_val"])
                )
