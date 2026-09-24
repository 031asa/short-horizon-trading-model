import unittest
import numpy as np
from utils.opening_three_action import select_actions


class ThreeActionTests(unittest.TestCase):
    def test_side_mirror_and_equality(self):
        self.assertEqual(select_actions(1, -1, .6, .3, .3), 2)
        self.assertEqual(select_actions(-1, 1, .3, .6, .3), 2)
        self.assertEqual(select_actions(1, -1, .51, .48, .05), 1)
        self.assertEqual(select_actions(1, 1, .2, .7, 0), 0)

    def test_flat_and_infinite_control(self):
        self.assertEqual(select_actions(1, 0, .4, .1, 0), 1)
        np.testing.assert_array_equal(select_actions([1, -1], [-1, -1], .7, .2, np.inf), [1, 0])

    def test_missing_not_imputed(self):
        with self.assertRaises(ValueError):
            select_actions(1, 1, np.nan, .5, .1)


if __name__ == '__main__':
    unittest.main()
