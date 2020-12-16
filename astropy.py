import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

"""
What forces are on each module?
    Gravitational force
    Thrust force (for delta V)
    Centrifugal force

Transfer Orbit
    Thrust

"""

current_path = os.path.dirname(__file__)

# Natural Constants
EARTH_RADIUS = 6378.140  # km
GRAVITY_CONST = 6.67430 * 10 ** -20  # km^3 kg^-1 s^-2
EARTH_MASS = 5.9722 * 10 ** 24  # kg
EARTH_MU = GRAVITY_CONST * EARTH_MASS


class Module:
    def __init__(self, start, thrust, mass, Ixx, Iyy, Ixy=0):
        """

        :param orbit_start: Starting orbit where module will transfer from
        :param target: Target orbit where module will transfer to.
        :param thrust: Thrust capability
        :param mass: Mass of module being transferred
        """
        self.x, self.y = start
        self.quad = self.get_quadrant()
        self.v = np.sqrt(EARTH_MU / np.sqrt(self.x**2 + self.y**2))
        self.a = 0
        self.T = thrust

        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Ixy = Ixy

        # Time step
        self.dt = 1  # seconds

    def calc_angle(self):

        # 0 - 180 = 0 - 180
        # 180 - 360 = -180 - 0
        theta = np.arctan2(self.y, self.x) * 180 / np.pi

        if abs(theta) > 90:
            theta = theta - 90
        elif theta < -90:
            theta = 180 - abs(theta)
        elif theta < 0:
            theta = 90 - abs(theta)
        return theta

    def get_quadrant(self):
        if self.x > 0 and self.y > 0:
            return 1
        elif self.x < 0 and self.y > 0:
            return 2
        elif self.x < 0 and self.y < 0:
            return 3
        elif self.x > 0 and self.y < 0:
            return 4
        return None

    def calc_tangential(self):
        if self.quad == 1:
            vx = self.v * np.cos(np.pi / 2 - self.calc_angle())
            vy = self.v * np.sin(np.pi / 2 - self.calc_angle())
            Tx = self.T * np.cos(np.pi / 2 - self.calc_angle())
            Ty = self.T * np.sin(np.pi / 2 - self.calc_angle())
        elif self.quad == 2:

        elif self.quad == 3:

        elif self.quad == 4:


    def calc_forces(self):

    def calc_acceleration(self):


    def calc_gravity(self):

    def calc_centrifugal(self):

    def calculate_orbit(self, target):
