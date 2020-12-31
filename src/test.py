import unittest
from src import astropy
from src import utils
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import PIL
import os
import random


class UtilsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.DataHandling = utils.DataHandling()
        self.AstroUtils = utils.AstroUtils()

    def test_path(self):
        self.assertIsInstance(self.DataHandling.data_path, str)
        self.assertEqual(self.DataHandling.data_path[-4], 'data')

    def test_import(self):
        planets = self.DataHandling.import_centre_body()
        for string in ['earth', 'Earth', 'eArtH', 'eaRTh']:
            earth = self.DataHandling.import_centre_body(string)
            self.assertEqual(earth['name'], 'Earth')
            self.assertEqual(earth['J2'], -1.082635854e-3)
        with self.assertRaises(AttributeError):
            var = self.DataHandling.import_centre_body(2)
        with self.assertRaises(ValueError):
            var = self.DataHandling.import_centre_body('Pluto')
        with self.assertRaises(KeyError):
            var = planets['M0on']
        with self.assertRaises(KeyError):
            var = planets['j2']

        self.assertEqual(planets['earth']['name'], 'Earth')
        self.assertEqual(planets['sun']['mass'], 1.989e30)
        self.assertEqual(planets['moon']['mu'], 4.282837e4)

    def test_save_figure(self):
        # Temporary matplotlib figure
        fig = plt.figure(figsize=(1, 1))

        # Test saving a non matplotlib figure
        with self.assertRaises(AttributeError):
            self.DataHandling.save_figure(2, 'hello')

        # Test saving normal image
        with self.assertRaises(AttributeError):
            img = PIL.Image.new('RGB', size=(1, 1))
            self.DataHandling.save_figure(img, 'test')

        # Test saving a matplotlib figure
        self.DataHandling.save_figure(fig, 'test')
        path_to_file = self.DataHandling.data_path + '/figures/test.jpg'
        self.assertTrue(os.path.exists(path_to_file))
        os.remove(path_to_file)

        # Test saving a png figure
        self.DataHandling.save_figure(fig, 'test', 'png')
        path_to_file = self.DataHandling.data_path + '/figures/test.png'
        self.assertTrue(os.path.exists(path_to_file))
        os.remove(path_to_file)

    def test_d2r(self):
        for n in np.arange(0.0, 180.0, 0.1):
            rad = self.AstroUtils.d2r(n)
            self.assertTrue(0 <= rad <= np.pi)
        with self.assertRaises(ValueError):
            rad = self.AstroUtils.d2r(random.uniform(-100000.0, -0.2))
        with self.assertRaises(ValueError):
            rad = self.AstroUtils.d2r(random.uniform(180.1, 100000.0))

    def test_r2d(self):
        for n in np.arange(0.0, np.pi, 0.01):
            deg = self.AstroUtils.r2d(n)
            self.assertTrue(0 <= deg <= 180.0)
        with self.assertRaises(ValueError):
            deg = self.AstroUtils.r2d(random.uniform(-100000.0, -0.2))
        with self.assertRaises(ValueError):
            deg = self.AstroUtils.r2d(random.uniform(np.pi + 0.01, 100000.0))

    def test_init_perts(self):
        var = self.AstroUtils.init_perts(True, False, False, False)
        self.assertTrue(var['J2'])
        self.assertFalse(var['aero'])
        self.assertFalse(var['moon_grav'])
        self.assertFalse(var['solar_grav'])

        var = self.AstroUtils.init_perts(False, True, False, False)
        self.assertFalse(var['J2'])
        self.assertTrue(var['aero'])
        self.assertFalse(var['moon_grav'])
        self.assertFalse(var['solar_grav'])

        var['moon_grav'] = True
        var['solar_grav'] = False
        self.assertTrue(var['moon_grav'])
        self.assertTrue(var['solar_grav'])

        var = self.AstroUtils.init_perts()
        self.assertFalse(var['moon_grav'])

    def test_plot_n_orbits(self):
        with self.assertRaises(TypeError):
            self.AstroUtils.plot_n_orbits(random.uniform(0, 100), ['test'])

        # Test input array is three dimensional
        with self.assertRaises(AttributeError):
            rand_list = random.sample(np.arange(-40000, 40000.0, 0.5), random.randint(0, 2))
            self.AstroUtils.plot_n_orbits([rand_list], ['test'])
        with self.assertRaises(AttributeError):
            rand_list = random.sample(np.arange(-40000, 40000.0, 0.5), random.randint(4, 7))
            self.AstroUtils.plot_n_orbits([rand_list], ['test'])

    def test_eci2perif(self):
        """
        Test the eci2perifocal coordinate conversion function. This requires analytical tests where random points are
        converted by hand and checked.
        :return:
        """
        for i in range(100):
            var = self.AstroUtils.eci2perif(random.uniform(0.0, np.pi),
                                            random.uniform(0.0, np.pi),
                                            random.uniform(0.0, np.pi))
            self.assertIsInstance(var, ndarray)
            self.assertEqual(np.shape(var), (3, 3))

        var = self.AstroUtils.eci2perif(0.667, 1.3, 1.0)
        self.assertEqual(np.around(var, 8).tolist(), [[-0.11189877, 0.57451883, 0.81080626],
                                                      [-0.84646044, -0.482533, 0.2250925],
                                                      [0.52056065, -0.66112784, 0.54030231]])

        var = self.AstroUtils.eci2perif(0.1, 0.1, 0.1)
        self.assertEqual(np.around(var, 8).tolist(), [[0.98011637, 0.19817307, 0.00996671],
                                                      [-0.19817307, 0.97512054, 0.09933467],
                                                      [0.00996671, -0.09933467, 0.99500417]])

    # TODO: finish writing tests for utils.py


class AstropyTest(unittest.TestCase):

    # TODO: Rewrite tests for astropy.py for new methods and/or functions.

    def test1(self):
        for i in range(100):
            Module_Test = astropy.Module(np.random.randint(-10000, 10000) * np.random.rand(2), 0, 0, 0, 0)
            self.assertGreaterEqual(Module_Test.get_angle(), 0)
            self.assertLessEqual(Module_Test.get_angle(), 2 * np.pi)


if __name__ == '__main__':
    unittest.main()
