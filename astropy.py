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
        :param start: Starting orbit where module will transfer from including radius of Earth, (x, y) coordinate.
        :param thrust: Thrust capability
        :param mass: Mass of module being transferred
        :param Ixx: Moment of inertia about x axis
        :param Iyy: Moment of inertia about y axis
        :param Ixy: Polar moment of inertia
        """
        # Current position in terms of x, y, theta, and radius (distance to Earth centre)
        self.x, self.y = start
        self.theta = self.get_angle()
        self.r = self.get_distance()
        # Current orbit quadrant
        self.quad = self.get_quadrant()
        # Initial velocity for circular orbit at position x, y
        # Velocity is tangential to orbit and orbit is counterclockwise
        self.v = np.sqrt(EARTH_MU / np.sqrt(self.x ** 2 + self.y ** 2))
        self.vx = None
        self.vy = None
        # Initial zero acceleration
        self.ax = 0
        self.ay = 0
        # Thrust capability
        self.T = thrust
        self.Tx = None
        self.Ty = None
        # Design parameters for object/module, mass and moments of inertia
        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Ixy = Ixy

        # Time step
        self.dt = 60  # seconds

        # Vector components
        self.calc_tangential()

    def get_angle(self):

        # 0 - 180 = 0 - 180
        # 180 - 360 = -180 - 0
        theta = np.arctan2(self.y, self.x) * 180 / np.pi

        if theta < 0:
            theta = 360 + theta
        return theta * np.pi / 180

    def get_distance(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def get_quadrant(self):
        if self.x >= 0 and self.y >= 0:
            return 1
        elif self.x < 0 and self.y >= 0:
            return 2
        elif self.x <= 0 and self.y < 0:
            return 3
        elif self.x > 0 and self.y < 0:
            return 4
        print('Corner case for quadrant found and not mitigated.')
        return None

    def calc_tangential(self):
        """
        Extract vector components x, y from tangential element i.e. for velocity and thrust (which both act in the
        same direction) This calculation is dependent on which quadrant the module is currently in, and assumes
        counterclockwise motion.
        :return: Updates current velocity and thrust in x and y direction.
        """
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
            vx = self.v * np.cos(3 * np.pi / 2 - self.theta)
            vy = -self.v * np.sin(3 * np.pi / 2 - self.theta)
            Tx = self.T * np.cos(3 * np.pi / 2 - self.theta)
            Ty = -self.T * np.sin(3 * np.pi / 2 - self.theta)
        else:
            vx = self.v * np.cos(2 * np.pi - self.theta)
            vy = self.v * np.sin(2 * np.pi - self.theta)
            Tx = self.T * np.cos(2 * np.pi - self.theta)
            Ty = self.T * np.sin(2 * np.pi - self.theta)

        self.vx, self.vy, self.Tx, self.Ty = vx, vy, Tx, Ty
        return

    def calc_gravity(self):
        """
        Calculate x and y components of the gravitational force towards Earth. Again this is dependent on the current
        quadrant.
        :return: x and y gravitational force components.
        """
        F = EARTH_MU * self.mass / self.r ** 2
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
        """
        Similar to gravity, but in opposing direction. Without thrust this should be equal and opposite to gravitation
        force.
        :return: x and y components of centrifugal force.
        """
        # Gives magnitude of the force without direction.
        F = self.mass * (self.v ** 2) / self.r
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
        """
        Calculate sum of forces in x and y direction. This includes gravitational, centrifugal, and thrust forces.
        :return: Resultant x and y forces on the module.
        """
        Fgx, Fgy = self.calc_gravity()
        Fcx, Fcy = self.calc_centrifugal()
        self.calc_tangential()
        Fx = Fgx + Fcx + self.Tx
        Fy = Fgy + Fcy + self.Ty
        return Fx, Fy

    def update_acc(self):
        """
        Update x and y acceleration due to resultant force.
        :return: updates ax and ay.
        """
        Fx, Fy = self.calc_forces()
        self.ax, self.ay = Fx / self.mass, Fy / self.mass
        return

    def update_velocity(self):
        self.vx += self.dt * self.ax
        self.vy += self.dt * self.ay
        self.v = np.sqrt(self.vx ** 2 + self.vy ** 2)
        return

    def update_pos(self):
        """
        Update x and y position of module based on current velocity in x and y direction. From this, the current
        distance, quadrant, and angle can be updated as well.
        :return: update
        """
        self.calc_tangential()
        self.x += self.dt * self.vx
        self.y += self.dt * self.vy
        self.quad = self.get_quadrant()
        self.theta = self.get_angle()
        self.r = self.get_distance()
        return

    def calculate_orbit(self, target):
        """
        Calculate Hohmann transfer to target orbit.
        :param target: Target altitude including radius of Earth
        :return: Trajectory path, in terms of x, y positions along this path.
        """
        path = [(self.x, self.y)]
        while self.r < target:
            self.update_acc()
            self.update_velocity()
            self.update_pos()
            path.append((self.x, self.y))
        return path


def plot_orbits(transfer_path, start_alt, target_alt, show_earth=True, filename=None):
    # Find path to data/figures folder
    data_path = os.path.relpath(f'../data/figures/{filename}', current_path)

    # Create figure and axes limits
    fig, ax = plt.subplots()
    # change default range so that new circles will work
    ax.set_xlim((-target_alt, target_alt))
    ax.set_ylim((-target_alt, target_alt))

    # Plot leo and target circular orbits
    leo = plt.Circle((0, 0), start_alt, fill=False, color='r')
    target = plt.Circle((0, 0), target_alt, fill=False, color='b')

    # Extract x and y coordinates from transfer path, then plot
    coords = [[i for i, j in transfer_path], [j for i, j in transfer_path]]
    ax.plot(coords[0], coords[1], color='black')

    # Create handles for the legend
    patches = [mpatches.Patch(color="red", label="LEO"), mpatches.Patch(color="blue", label="Target")]

    # If you want to see Earth, it is plotted here
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
    leo_alt = 1750 + EARTH_RADIUS
    geo_alt = 35786 + EARTH_RADIUS
    transfer = Module((leo_alt, 0), -5, 100e3, 50, 50)
    transfer2 = Module((leo_alt, 0), 0, 100e3, 50, 50)
    path = transfer.calculate_orbit(geo_alt)
    path2 = transfer2.calculate_orbit(geo_alt)
    plot_orbits(path, leo_alt, geo_alt)
    plot_orbits(path2, leo_alt, geo_alt)
    print('Done')
