from sample import hoge
import unittest


class TestHoge(unittest.TestCase):
    """hoge package に含まれる関数のテストを行う。
    """

    def test_one(self):
        self.assertTrue(hoge.check_equal(1, 1))
