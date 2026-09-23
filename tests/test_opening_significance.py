import unittest
from scripts.check_three_significance import flip_p

class SignFlipTests(unittest.TestCase):
    def test_exact_extremes(self):
        self.assertEqual(flip_p([1,1,1]),.25)
        self.assertEqual(flip_p([1,1,1,1]),.125)
        self.assertEqual(flip_p([1,-1]),1.)

    def test_scale_and_sign_invariance(self):
        self.assertEqual(flip_p([1,-2,4]),flip_p([-10,20,-40]))

if __name__=='__main__':unittest.main()
