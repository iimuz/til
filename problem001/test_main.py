import unittest

from .main import missing_number


class TestMain(unittest.TestCase):
    """ test class of main.py
    """

    def test_example(self) -> None:
        """
        """
        test_case = [
            {"array": [1, 2, 3, 4, 6], "n": 6, "exp": {5}},
            {"array": [1, 2, 3, 4, 6, 7, 9, 8, 10], "n": 10, "exp": {5}},
            {"array": [1, 2, 3, 4, 6, 9, 8], "n": 10, "exp": {5, 7, 10}},
            {"array": [1, 2, 3, 4, 9, 8], "n": 10, "exp": {5, 6, 7, 10}},
        ]
        for params in test_case:
            with self.subTest(params=params):
                self.assertSetEqual(
                    params["exp"], missing_number(params["array"], params["n"])
                )


if __name__ == "__main__":
    unittest.main()
