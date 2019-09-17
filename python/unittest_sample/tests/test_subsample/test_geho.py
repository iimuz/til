from sample.subsample import geho
import unittest


class TestGeho(unittest.TestCase):
    """geho package に含まれる関数のテストを行う。
    """

    def test_one(self):
        self.assertEqual(geho.add_three(1), 4)
