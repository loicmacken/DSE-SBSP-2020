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
        self.theta = self.get_angle()
        self.r = self.get_distance()
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

    def get_angle(self):

        # 0 - 180 = 0 - 180
        # 180 - 360 = -180 - 0
        theta = np.arctan2(self.y, self.x) * 180 / np.pi

        if theta < 0:
            theta = 360 + theta
        return theta * np.pi / 180

    def get_distance(self):
        return np.sqrt(self.x**2 + self.y**2)

    def get_quadrant(self):
        if self.x >= 0 and self.y > 0:
            return 1
        elif self.x < 0 and self.y >= 0:
            return 2
        elif self.x < 0 and self.y < 0:
            return 3
        elif self.x > 0 and self.y < 0:
            return 4
        return None

    def calc_tangential(self):
        if self.quad == 1:
            vx = self.v * np.cos(np.pi / 2 - self.theta)
            vy = self.v * np.sin(np.pi / 2 - self.theta)
            Tx = self.T * np.cos(np.pi / 2 - self.theta)
            Ty = self.T * np.sin(np.pi / 2 - self.theta)
        elif self.quad == 2:
            vx = -self.v * np.sin(np.pi - self.theta)
            vy = -self.v * np.cos(np.pi - self.theta)
            Tx = -self.T * np.sin(np.pi - self.theta)
            Ty = -self.T * np.cos(np.pi - self.theta)
        elif self.quad == 3:
            vx = self.v * np.sin(3 * np.pi / 2 - self.theta)
            vy = -self.v * np.cos(3 * np.pi / 2 - self.theta)
            Tx = self.T * np.sin(3 * np.pi / 2 - self.theta)
            Ty = -self.T * np.cos(3 * np.pi / 2 - self.theta)
        else:
            vx = self.v * np.cos(2 * np.pi - self.theta)
            vy = self.v * np.sin(2 * np.pi - self.theta)
            Tx = self.T * np.cos(2 * np.pi - self.theta)
            Ty = self.T * np.sin(2 * np.pi - self.theta)

        return vx, vy, Tx, Ty

    def calc_gravity(self):
        F = EARTH_MU * self.mass / self.get_distance() ** 2
        if self.quad == 1:
            Fx = -F * np.cos(self.theta)
            Fy = -F * np.sin(self.theta)
        elif self.quad == 2:
            Fx = F * np.cos(np.pi - self.theta)
            Fy = -F * np.sin(np.pi - self.theta)
        elif self.quad == 3:
            Fx = F * np.sin(3 * np.pi / 2 - self.theta)
            Fy = F * np.cos(3 * np.pi / 2 - self.theta)
        else:
            Fx = -F * np.cos(2 * np.pi - self.theta)
            Fy = F * np.sin(2 * np.pi - self.theta)

        return Fx, Fy

    def calc_centrifugal(self):
        F = self.mass * (self.v ** 2) / self.get_distance()
        if self.quad == 1:
            Fx = F * np.cos(self.theta)
            Fy = F * np.sin(self.theta)
        elif self.quad == 2:
            Fx = -F * np.cos(np.pi - self.theta)
            Fy = F * np.sin(np.pi - self.theta)
        elif self.quad == 3:
            Fx = -F * np.sin(3 * np.pi / 2 - self.theta)
            Fy = -F * np.cos(3 * np.pi / 2 - self.theta)
        else:
            Fx = F * np.cos(2 * np.pi - self.theta)
            Fy = -F * np.sin(2 * np.pi - self.theta)

        return Fx, Fy

    def calc_forces(self):
        Fgx, Fgy = self.calc_gravity()
        Fcx, Fcy = self.calc_centrifugal()
        vx, vy, Tx, Ty = self.calc_tangential()
        Fx = Fgx + Fcx + Tx
        Fy = Fgy + Fcy + Ty
        return Fx, Fy

    def calc_acceleration(self):
        Fx, Fy = self.calc_forces()
        ax, ay = Fx / self.mass, Fy / self.mass
        self.a = np.sqrt(ax**2 + ay**2)
        return ax, ay

    def update_velocity(self):
        self.v += self.dt * self.a
        return

    def update_pos(self):
        vx, vy, Tx, Ty = self.calc_tangential()
        self.x += self.dt * vx
        self.y += self.dt * vy
        self.quad = self.get_quadrant()
        self.theta = self.get_angle()
        self.r = self.get_distance()
        return

    def calculate_orbit(self, target):
