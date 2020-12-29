import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
from src.utils import *

current_path = os.path.dirname(__file__)

data_root = str(get_data_root())

plt.style.use('dark_background')

# Natural Constants
EARTH_RADIUS = 6378.140  # km
GRAVITY_CONST = 6.67430 * 10 ** -20  # km^3 kg^-1 s^-2
EARTH_MASS = 5.9722 * 10 ** 24  # kg
EARTH_MU = GRAVITY_CONST * EARTH_MASS
g0 = 9.81


class Module:
    def __init__(self, start, thrust, isp, mass, Ixx, Iyy, Ixy=0, dt=1):
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
        self.isp = isp
        # Design parameters for object/module, mass and moments of inertia
        self.mass = mass
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Ixy = Ixy

        # Time step
        self.dt = dt  # seconds

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
        Fx = Fgx + self.Tx
        Fy = Fgy + self.Ty
        return round(Fx, 2), round(Fy, 2)

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
        self.calc_tangential()
        return

    def update_pos(self):
        """
        Update x and y position of module based on current velocity in x and y direction. From this, the current
        distance, quadrant, and angle can be updated as well.
        :return: update
        """
        self.x += self.dt * self.vx
        self.y += self.dt * self.vy
        self.quad = self.get_quadrant()
        self.theta = self.get_angle()
        self.r = self.get_distance()
        return

    def update_mass(self):
        self.mass += self.dt * self.T / (self.isp * g0)
        return

    def calculate_orbit(self, target):
        """
        Calculate Hohmann transfer to target orbit.
        :param target: Target altitude including radius of Earth
        :return: Trajectory path, in terms of x, y positions along this path.
        """
        path = [(self.x, self.y)]
        t = 0
        t_end = 100000
        while self.r < target and t < t_end:
            self.update_acc()
            self.update_velocity()
            self.update_pos()
            if self.r <= EARTH_RADIUS:
                print("Oops you crashed.")
                break
            elif t == t_end:
                print("Time's up")
            path.append((self.x, self.y))
            t += self.dt
        return path


def plot_orbits(transfer_path, start_alt, target_alt, show_earth=True, filename=None):
    """
    Plot the trajectory path of the module from parking orbit to target orbit.
    :param transfer_path: List of x, y cooardinates of the trajectory path.
    :param start_alt: Starting altitude including the Earth's radius.
    :param target_alt: Target altitude including the Earth's radius.
    :param show_earth: Show the Earth in the plot, boolean.
    :param filename: Name of plot if you want to save it to data/figures.
    :return: Trajectory plot.
    """
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


class OrbitPropagator:
    """
    Inspired by Alfonso Gonzalez
    Orbit propagator class which calculates orbit propagation of an orbital body.
    """

    def __init__(self, r0, v0, tspan, dt, cb=None):
        if cb is None:
            cb = import_data('earth')
        self.r0 = r0
        self.v0 = v0

        self.tspan = tspan
        self.dt = dt
        self.n_steps = int(np.ceil(self.tspan / self.dt))

        self.ys = np.zeros((self.n_steps, 6))
        self.ts = np.zeros((self.n_steps, 1))
        # Initial conditions
        self.y0 = self.r0 + self.v0
        self.ys[0] = self.y0
        self.ts[0] = 0
        self.step = 1
        self.rs = None
        self.vs = None
        self.cb = cb

    def propagate_orbit(self):
        # Initiate solver
        solver = ode(self.diffy_q)
        solver.set_integrator('lsoda')
        solver.set_initial_value(self.y0, 0)

        # Propogate orbit
        while solver.successful() and self.step < self.n_steps:
            solver.integrate(solver.t + self.dt)
            self.ts[self.step] = solver.t
            self.ys[self.step] = solver.y
            self.step += 1
        self.rs = self.ys[:, :3]
        self.vs = self.ys[:, 3:]
        return

    def diffy_q(self, t, y):
        # Unpack state space
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])

        # Norm of the radius vector
        norm_r = np.linalg.norm(r)

        # Two body acceleration
        ax, ay, az = -r * self.cb['mu'] / norm_r ** 3

        return [vx, vy, vz, ax, ay, az]

    def plot_3d(self, show_plot=False, save_plot=False, title='Test Title'):
        fig = plt.figure(figsize=(14, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(self.rs[:, 0], self.rs[:, 1], self.rs[:, 2], 'b', label='Trajectory')
        ax.plot([self.rs[0, 0]], [self.rs[0, 1]], [self.rs[0, 2]], 'bo', label='Initial Position')

        # Plot central body
        _u, _v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        _x = self.cb['radius'] * np.cos(_u) * np.sin(_v)
        _y = self.cb['radius'] * np.sin(_u) * np.sin(_v)
        _z = self.cb['radius'] * np.cos(_v)
        ax.plot_surface(_x, _y, _z, cmap='Greens')

        # Plot the x,y,z vectors
        l = self.cb['radius'] * 2
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]

        ax.quiver(x, y, z, u, v, w, color='r')

        max_val = np.max(np.abs(self.rs))

        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])

        ax.set_xlabel(['X (km)'])
        ax.set_ylabel(['Y (km)'])
        ax.set_zlabel(['Z (km)'])

        ax.set_aspect('auto')

        ax.set_title(title)

        plt.legend()

        if show_plot:
            plt.show()
        else:
            plt.cla()
            plt.clf()
            plt.close()

        if save_plot:
            save_path = data_root + '/figures'
            plt.savefig(save_path + f"/{self.cb['name']}_init_{round(self.rs[0][0])}_fin_{round(self.rs[-1][0])}.png",
                        dpi=300)
        return


if __name__ == "__main__":
    # Choose orbital body to get data from
    data = import_data('earth')

    # First orbiting body
    r_mag = data['radius'] + 1000.0
    v_mag = np.sqrt(data['mu'] / r_mag)
    r0 = [r_mag, r_mag*0.01, 0]
    v0 = [0, v_mag, v_mag*0.5]

    # Second orbiting body
    r_mag = data['radius'] + 408.0
    v_mag = np.sqrt(data['mu'] / r_mag)
    r00 = [r_mag, 0, 0]
    v00 = [0, v_mag, 0]

    # Third orbiting body
    r_mag = data['radius'] + 1800
    v_mag = np.sqrt(data['mu'] / r_mag)
    r000 = [-r_mag, -r_mag*0.01, 0]
    v000 = [0, -v_mag, -v_mag*0.3]

    orbit0 = OrbitPropagator(r0, v0,
                             tspan=3600 * 10,
                             dt=10.0,
                             cb=data)
    orbit1 = OrbitPropagator(r00, v00,
                             tspan=3600 * 10,
                             dt=10.0,
                             cb=data)
    orbit2 = OrbitPropagator(r000, v000,
                             tspan=3600 * 10,
                             dt=10.0,
                             cb=data)

    orbit0.propagate_orbit()
    orbit1.propagate_orbit()
    orbit2.propagate_orbit()

    plot_n_orbits([orbit0.rs, orbit1.rs, orbit2.rs],
                  labels=['Random Orbit 1', 'ISS Orbit', 'Random Orbit 2'],
                  show_plot=True,
                  save_plot=False,
                  title='Randomly initiated orbit with ISS')

    # orbit0.plot_3d(show_plot=True, save_plot=True, title="Earth with geostationary orbit")

    """

    transfer = Module((leo_alt, 0), 5, 4800, 500e3, 50, 50)
    # transfer2 = Module((leo_alt, 0), 0, 100e3, 50, 50)
    # transfer3 = Module((leo_alt, 0), 5, 100e3, 50, 50)

    path = transfer.calculate_orbit(geo_alt)
    # path2 = transfer2.calculate_orbit(geo_alt)
    # path3 = transfer3.calculate_orbit(geo_alt)
    plot_orbits(path, leo_alt, geo_alt)
    # plot_orbits(path2, leo_alt, geo_alt)
    # plot_orbits(path3, leo_alt, geo_alt)
    print('Done')

    """
