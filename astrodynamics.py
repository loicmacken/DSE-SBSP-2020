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

### Mission Parameters ####
# Low Earth Orbit altitude from surface
LEO_ALT = 1750
# Altitude from earth centre
h_leo = LEO_ALT + EARTH_RADIUS
# Target orbit
h_target = 35800 + EARTH_RADIUS


class Transfer:
    def __init__(self, leo, target, thrust, mass):
        self.leo = leo
        self.target = target
        self.mass = mass
        self.i_xx = 0
        self.i_yy = 0
        self.i_xy = 0

        self.x = self.leo
        self.y = 0
        # Angle of orbit
        self.theta = self.calculate_angle()

        self.t = 0
        self.dt = 1  # seconds

        self.v = np.sqrt(EARTH_MU / self.leo)
        self.v_x, self.v_y = self.vectorize(self.v, tang=True)

        self.r = np.sqrt(self.x ** 2 + self.y ** 2)

        # Orbital constants
        self.Fg = -EARTH_MU * self.mass / self.r ** 2

        # Check this vectorisation
        self.Fg_x, self.Fg_y = self.vectorize(self.Fg)

        self.T = thrust
        self.Tx, self.Ty = self.vectorize(self.T, tang=True)

        self.Fc = self.mass * (self.v ** 2) / self.r
        self.Fc_x, self.Fc_y = self.vectorize(self.Fc)

        self.Fx = self.Fg_x + self.Tx
        self.Fy = self.Fg_y + self.Ty

        self.ax = self.Fx / self.mass
        self.ay = self.Fy / self.mass

    def calculate_angle(self):
        return np.arctan2(self.y, self.x) * np.pi / 180



    def vectorize(self, F, tang=False):
        if tang:
            Fx = F * np.cos(np.pi / 2 - self.theta)
            Fy = F * np.sin(np.pi / 2 - self.theta)
        else:
            Fx = np.cos(self.theta) * F
            Fy = np.sin(self.theta) * F

        return Fx, Fy

    def update_pos(self):
        self.x = self.x + self.dt * self.v_x
        self.y = self.y + self.dt * self.v_y
        self.r = np.sqrt(self.x ** 2 + self.y ** 2)
        self.theta = np.arctan2(self.y, self.x) * np.pi / 180

    def update_velocity(self):
        self.v_x = self.v_x + self.dt * self.ax
        self.v_y = self.v_y + self.dt * self.ay
        self.v = np.sqrt(self.v_x ** 2 + self.v_y ** 2)

    def update_forces(self):
        self.Fc = self.mass * (self.v ** 2) / self.r
        self.Fc_x, self.Fc_y = self.vectorize(self.Fc)

        self.Fg = EARTH_MU * self.mass / self.r ** 2
        self.Fg_x, self.Fg_y = self.vectorize(self.Fg)

        self.Tx, self.Ty = self.vectorize(self.T, tang=True)

        self.Fx = self.Fg_x + self.Tx + self.Fc_x
        self.Fy = self.Fg_y + self.Ty + self.Fc_y


    def update(self):
        # self.mass = self.mass - self.impulse * self.dt
        self.update_pos()
        self.update_velocity()
        self.update_forces()

        self.ax = self.Fx / self.mass
        self.ay = self.Fy / self.mass

    def constant_thrust_spiral(self):
        """
        Calculate Hohmann transfer with constant thrust.
        :return:
        """
        path = list()
        # This needs to be worked out, because velocity should go down with higher orbits but we increase thrust?
        while self.x < self.target:
            path.append((self.x, self.y))
            self.update()

        return path


def plot_orbits(show_earth=True, filename=None):
    data_path = os.path.relpath(f'../data/figures/{filename}', current_path)

    fig, ax = plt.subplots()
    # change default range so that new circles will work
    ax.set_xlim((-h_target, h_target))
    ax.set_ylim((-h_target, h_target))

    leo = plt.Circle((0, 0), h_leo, fill=False, color='r')
    target = plt.Circle((0, 0), h_target, fill=False, color='b')

    patches = [mpatches.Patch(color="red", label="LEO"), mpatches.Patch(color="blue", label="Target")]

    if show_earth:
        earth = plt.Circle((0, 0), EARTH_RADIUS, color='g')
        patches.append(mpatches.Patch(color='green', label='Earth'))
        ax.add_artist(earth)

    ax.add_artist(leo)
    ax.add_artist(target)
    ax.legend(handles=patches)
    ax.grid()
    if filename:
        fig.savefig(data_path)

    plt.show()


if __name__ == "__main__":
    mod_transfer = Transfer(h_leo, h_target, 5, 1, 100e3)
    path = mod_transfer.constant_thrust_spiral()

    plot_orbits()
