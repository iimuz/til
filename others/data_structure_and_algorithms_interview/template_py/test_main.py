import unittest

from .main import run


class TestTemplate(unittest.TestCase):
    def test_example(self) -> None:
        test_case = [{"expected": 0}]
        for params in test_case:
            with self.subTest(params=params):
                self.assertEqual(params["expected"], run())
