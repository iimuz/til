import unittest

from .main import search_duplicate_number


class TestSearchDuplicate(unittest.TestCase):
    """ 重複アイテムを探索する関数のテスト
    """

    def test_example(self) -> None:
        """
        """
        test_case = [
            {"arr": [1, 1, 2, 2, 3, 4, 5], "expected": {1, 2}},
            {"arr": [1, 1, 1, 1, 1, 1, 1], "expected": {1}},
            {"arr": [1, 2, 3, 4, 5, 6, 7], "expected": set},
            {"arr": [1, 2, 1, 1, 1, 1, 1], "expected": {1}},
        ]
        for params in test_case:
            with self.subTest(params=params):
                self.assertSetEqual(
                    params["expected"], search_duplicate_number(params["arr"])
                )


if __name__ == "__main__":
    unittest.main()
