import sample
import unittest


class TestSample(unittest.TestCase):
    """sample に含まれる関数のテストを行う。
    """

    def test_one(self):
        self.assertEqual(4, sample.add3(1))
