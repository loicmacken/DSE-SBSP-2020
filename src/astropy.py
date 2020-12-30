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
cb_data = DataHandling().import_centre_body()
EARTH = cb_data['earth']
EARTH_RADIUS = EARTH['radius']  # km
GRAVITY_CONST = 6.67430 * 10 ** -20  # km^3 kg^-1 s^-2
EARTH_MASS = EARTH['mass']  # kg
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



class OrbitPropagator(AstroUtils):
    """
    Inspired by Alfonso Gonzalez
    Orbit propagator class which calculates orbit propagation of an orbital body.
    """

    def __init__(self, state0, tspan, dt, coes=False, deg=True, cb=None, perts=None):
        """
        Initialise orbit instance, inheriting astrodynamics tools from AstroUtils.
        :param state0: Initial state conditions;
                        if coes=True: [Altitude [km], Eccentricity, Inclination, True Anomaly,
                        Argument of Perigee, Right Ascension of Ascending Node]
                        if coes=False: [Initial position [x, y, z], Initial velocity [u, v, w]]
        :param tspan: Time span
        :param dt: Time step
        :param coes: Classical orbital elements or simply r0 and v0
        :param cb: Centre body data, if None then Earth will be used
        """
        if perts is None:
            perts = self.init_perts()
        if cb is None:
            cb = DataHandling().import_centre_body('earth')
        if coes:
            self.r0, self.v0 = self.coes2rv(state0, deg=deg, mu=cb['mu'])
        else:
            self.r0 = state0[:3]
            self.v0 = state0[3:]

        self.tspan = tspan
        self.dt = dt
        self.n_steps = int(np.ceil(self.tspan / self.dt))

        # Initialise arrays, memory allocation
        self.ys = np.zeros((self.n_steps, 6))
        self.ts = np.zeros((self.n_steps, 1))
        self.coes = np.zeros((self.n_steps, 6))

        # Initial conditions
        self.y0 = self.r0.tolist() + self.v0.tolist()
        self.ys[0] = self.y0
        self.ts[0] = 0
        self.step = 1
        self.rs = self.ys[:, :3]
        self.vs = self.ys[:, 3:]
        self.cb = cb

        self.perts = perts

    def propagate_orbit(self):
        """
        Use scipy ode solver to propogate orbit through integration of position and velocity over time.
        :return: Updates orbit instance's r values over time steps
        """
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
        """
        Differential equation defining the orbit propagation.
        :param t: Current time (used in ode solver)
        :param y: Current state space
        :return: New velocities and accelerations
        """
        # Unpack state space
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])

        # Norm of the radius vector
        norm_r = np.linalg.norm(r)

        # Two body acceleration
        a = -r * self.cb['mu'] / norm_r ** 3

        if self.perts['J2']:
            z2 = r[2] ** 2
            r2 = norm_r ** 2
            tx = r[0] / norm_r * (5 * z2 / r2 - 1)
            ty = r[1] / norm_r * (5 * z2 / r2 - 1)
            tz = r[2] / norm_r * (5 * z2 / r2 - 3)

            a_j2 = 1.5 * self.cb['J2'] * self.cb['mu'] * self.cb['radius'] ** 2 / norm_r ** 4 * np.array([tx, ty, tz])

            a += a_j2

        return [vx, vy, vz, a[0], a[1], a[2]]

    def calculate_coes(self):
        """
        :return: Updates COEs array using rv2coes util function with existing rs and vs from propagation.
        """
        print('Calculating Classical Orbital Elements (COEs)...')

        for n in range(self.n_steps):
            self.coes[n, :] = self.rv2coes(self.rs[n, :], self.vs[n, :], mu=self.cb['mu'], deg=True)

    def plot_coes(self, hours=False, days=False, show_plot=False, save_plot=False, title='COEs'):
        """
        Order of COEs: a, e_norm, i, ta, aop, raan
        :param hours: Show time elapsed in hours
        :param days: Show time elapsed in days
        :param show_plot: Show plot
        :param save_plot: Save plot
        :param title: Plot title
        :return:
        """
        print("Plotting Classical Orbital Elements (COEs)...")
        # Create figure and axes
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))
        # Figure title
        fig.suptitle(title, fontsize=20)

        # x axis
        if hours:
            ts = self.ts / 3600.0
            xlabel = 'Time Elapsed (hours)'
        elif days:
            ts = self.ts / 3600.0 / 24.0
            xlabel = 'Time Elapsed (days)'
        else:
            ts = self.ts
            xlabel = 'Time Elapsed (seconds)'

        # Plot true anomaly
        axs[0, 0].plot(ts, self.coes[:, 3])
        axs[0, 0].set_title('True Anomaly vs Time')
        axs[0, 0].grid(True)
        axs[0, 0].set_ylabel('Angle (degrees)')
        axs[0, 0].set_xlabel(xlabel)

        # Plot semi-major axis
        axs[1, 0].plot(ts, self.coes[:, 0])
        axs[1, 0].set_title('Semi-Major Axis vs Time')
        axs[1, 0].grid(True)
        axs[1, 0].set_ylabel('Semi-Major Axis (km)')
        axs[1, 0].set_xlabel(xlabel)

        # Plot eccentricity
        axs[0, 1].plot(ts, self.coes[:, 1])
        axs[0, 1].set_title('Eccentricity vs Time')
        axs[0, 1].grid(True)

        # Plot argument of periapsis
        axs[0, 2].plot(ts, self.coes[:, 4])
        axs[0, 2].set_title('Argument of Periapsis vs Time')
        axs[0, 2].grid(True)
        axs[0, 2].set_ylabel('Angle (degrees)')
        axs[0, 2].set_xlabel(xlabel)

        # Plot inclination
        axs[1, 1].plot(ts, self.coes[:, 2])
        axs[1, 1].set_title('Inclination vs Time')
        axs[1, 1].grid(True)
        axs[1, 1].set_ylabel('Angle (degrees)')
        axs[1, 1].set_xlabel(xlabel)

        # Plot RAAN
        axs[1, 2].plot(ts, self.coes[:, 5])
        axs[1, 2].set_title('RAAN vs Time')
        axs[1, 2].grid(True)
        axs[1, 2].set_ylabel('Angle (degrees)')
        axs[1, 2].set_xlabel(xlabel)

        if show_plot:
            plt.show()
        if save_plot:
            DataHandling().save_figure(fig, title.replace(' ', '_'))

    # def plot_3d(self, show_plot=False, save_plot=False, title='Test Title'):
    #     fig = plt.figure(figsize=(10, 10))
    #     ax = fig.add_subplot(111, projection='3d')
    #
    #     # Plot trajectory
    #     ax.plot(self.rs[:, 0], self.rs[:, 1], self.rs[:, 2], 'b', label='Trajectory')
    #     ax.plot([self.rs[0, 0]], [self.rs[0, 1]], [self.rs[0, 2]], 'bo', label='Initial Position')
    #
    #     # Plot central body
    #     _u, _v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    #     _x = self.cb['radius'] * np.cos(_u) * np.sin(_v)
    #     _y = self.cb['radius'] * np.sin(_u) * np.sin(_v)
    #     _z = self.cb['radius'] * np.cos(_v)
    #     ax.plot_surface(_x, _y, _z, cmap='Greens')
    #
    #     # Plot the x,y,z vectors
    #     l = self.cb['radius'] * 2
    #     x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    #     u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]
    #
    #     ax.quiver(x, y, z, u, v, w, color='r')
    #
    #     max_val = np.max(np.abs(self.rs))
    #
    #     ax.set_xlim([-max_val, max_val])
    #     ax.set_ylim([-max_val, max_val])
    #     ax.set_zlim([-max_val, max_val])
    #
    #     ax.set_xlabel('X (km)')
    #     ax.set_ylabel('Y (km)')
    #     ax.set_zlabel('Z (km)')
    #
    #     ax.set_aspect('auto')
    #
    #     ax.set_title(title)
    #
    #     plt.legend()
    #
    #     if show_plot:
    #         plt.show()
    #     else:
    #         plt.cla()
    #         plt.clf()
    #         plt.close()
    #
    #     if save_plot:
    #         save_path = data_root + '/figures'
    #         fig.savefig(save_path + f"/{title}.jpg")
    #     return


if __name__ == "__main__":
    # Choose orbital body to get data from
    data = cb_data['earth']

    # ISS
    # Altitude, Eccentricity, Inclination, True anomaly, Argument of Perigee, Right Acsension of Ascending Node

    c0 = [data['radius'] + 414.0, 0.0000867, 51.6417, 0.0, 165.1519, 95.4907]

    # Geostationary
    c1 = [data['radius'] + 35786, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Random
    c2 = [data['radius'] + 1750, 0.0, 0.0, 0.0, 0.0, 0.0]

    tspan = 3600 * 24 * 2

    # Creat instances
    o0 = OrbitPropagator(c0,
                         tspan=tspan,
                         dt=10.0,
                         coes=True,
                         cb=data,
                         perts=AstroUtils.init_perts(J2=True))
    o1 = OrbitPropagator(c1,
                         tspan=tspan,
                         dt=10.0,
                         coes=True,
                         cb=data,
                         perts=AstroUtils.init_perts(J2=True))
    o2 = OrbitPropagator(c2,
                         tspan=tspan,
                         dt=10.0,
                         coes=True,
                         cb=data,
                         perts=AstroUtils.init_perts(J2=True))
    o0.propagate_orbit()
    o1.propagate_orbit()
    o2.propagate_orbit()

    o2.calculate_coes()
    # o2.plot_coes(hours=True, show_plot=False, save_plot=False)

    AstroUtils.plot_n_orbits([o0.rs, o1.rs, o2.rs],
                             labels=['ISS Orbit', 'Geostationary Orbit', 'IKAROS orbit'],
                             show_plot=True,
                             save_plot=True,
                             title=f'Orbit comparisons with J2 perturbation - {round(o0.ts[-1][0] / 3600)} hrs')

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
