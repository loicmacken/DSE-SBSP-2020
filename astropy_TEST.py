import unittest
import astropy
import numpy as np

class astropy_TEST(unittest.TestCase):

    def setUp(self):
        # self.Module_Test = astropy.Module(0,0,0,0,0)
        pass

    def test1(self):
        for i in range(100):
            Module_Test = astropy.Module(np.random.randint(-10000, 10000) * np.random.rand(2),0,0,0,0)
            self.assertGreaterEqual(Module_Test.calc_angle(), 0)
            self.assertLessEqual(Module_Test.calc_angle(), 360)
        

if __name__ == '__main__':
    unittest.main()