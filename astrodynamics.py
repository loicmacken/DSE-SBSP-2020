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
GRAVITY_CONST = 6.67430 * 10 ** -11  # m^3 kg^-1 s^-2
EARTH_MASS = 5.9722 * 10 ** 24  # kg
MODULE_MASS = 500000  # kg

### Mission Parameters ####
# Low Earth Orbit altitude from surface
LEO_ALT = 1750
# Altitude from earth centre
h_leo = LEO_ALT + EARTH_RADIUS
# Target orbit
h_target = 35800 + EARTH_RADIUS

DELTA_V = 3.44  # km/s

### Design Parameters ###
vt = False

m_module = 3600
i_xx = 0
i_yy = 0
i_xy = 0


class Transfer:
    def __init__(self, leo, target, thrust, impulse, mass):
        self.leo = leo
        self.target = target
        self.mass = mass

        self.x = self.leo
        self.y = 0

        self.t = 0
        self.dt = 3600  # seconds

        self.v = np.sqrt(GRAVITY_CONST / self.leo)
        self.v_x = self.v * np.cos(np.pi / 2 - self.theta)
        self.v_y = self.v * np.sin(np.pi / 2 - self.theta)

        self.r = np.sqrt(self.x ** 2 + self.y ** 2)

        # Orbital constants
        self.Fg = GRAVITY_CONST * EARTH_MASS * self.mass / self.r ** 2

        # Angle of orbit
        self.theta = np.arctan2(self.y, self.x) * np.pi / 180
        # Check this vectorisation
        self.Fg_x = np.cos(self.theta) * self.Fg
        self.Fg_y = np.sin(self.theta) * self.Fg

        self.T = thrust
        self.Tx = self.T * np.cos(np.pi / 2 - self.theta)
        self.Ty = self.T * np.sin(np.pi / 2 - self.theta)

        self.impulse = impulse

        self.Fc = self.mass * (self.v ** 2) / self.r

        self.Fc_x = np.cos(self.theta) * self.Fc
        self.Fc_y = np.sin(self.theta) * self.Fc

        self.Fx = self.Fg_x + self.Tx + self.Fc_x
        self.Fy = self.Fg_y + self.Ty + self.Fc_y

        self.ax = self.Fx / self.mass
        self.ay = self.Fy / self.mass

    def update(self):
        self.mass = self.mass - self.impulse * self.dt

        self.v_x = self.v_x + self.dt * self.ax
        self.v_y = self.v_y + self.dt * self.ay
        self.v = np.sqrt(self.v_x ** 2 + self.v_y ** 2)

        self.Fc = self.mass * (self.v ** 2) / self.r
        self.Fc_x = np.cos(self.theta) * self.Fc
        self.Fc_y = np.sin(self.theta) * self.Fc

        self.ax = self.Fx / self.mass
        self.ay = self.Fy / self.mass

    def constant_thrust_spiral(self):
        """
        Calculate Hohmann transfer with constant thrust.
        :return:
        """
        path = list()
        while self.v > np.sqrt(GRAVITY_CONST / self.target):
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


#
# sc_vx = 0
# sc_vy = np.sqrt(G * me / h1)
#
#
# def gravity(sc_x, sc_y, r):
#     f = (G * me * m) / r ** 2
#     fx = - sc_x / r * f
#     fy = - sc_y / r * f
#     return fx, fy
#
#
# def thrust(sc_x, sc_y, r):
#     if sc_x >= 0 and sc_y >= 0:
#         vtx = -1
#         vty = 1
#     elif sc_x < 0 and sc_y >= 0:
#         vtx = -1
#         vty = -1
#     elif sc_x >= 0 and sc_y < 0:
#         vtx = 1
#         vty = 1
#     else:
#         vtx = 1
#         vty = -1
#     if vt:
#         tx = vtx * abs(sc_y / r * t)
#         ty = vty * abs(sc_x / r * t)
#     else:
#         tx = -vtx * abs(sc_y / r * t)
#         ty = -vty * abs(sc_x / r * t)
#     return tx, ty

if __name__ == "__main__":
    plot_orbits()
