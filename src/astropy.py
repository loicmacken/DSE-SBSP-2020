import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.integrate import ode
from mpl_toolkits.mplot3d import Axes3D
from src.utils import *
from src.utils import AstroUtils as au
import warnings

data_root = str(get_data_root())

plt.style.use('dark_background')

# Natural Constants
GRAVITY_CONST = 6.67430 * 10 ** -20  # km^3 kg^-1 s^-2
cb_data = DataHandling().import_centre_body()
EARTH = cb_data['earth']
EARTH_RADIUS = EARTH['radius']  # km
EARTH_MASS = EARTH['mass']  # kg
EARTH_MU = EARTH['mu']  # km^3 s^-2


class Error(Exception):
    pass


class CrashError(Error):

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class OrbitPropagator(AstroUtils):
    """
    Inspired by Alfonso Gonzalez
    Orbit propagator class which calculates orbit propagation of an orbital body.
    """

    def __init__(self, state0, tspan, dt, target=None, coes=False, deg=True, mass0=0.0, cb=None, perts=None):
        """
        Initialise orbit instance, inheriting astrodynamics tools from AstroUtils.
        :param state0: Initial state conditions;
                        if coes=True: [Altitude [km], Eccentricity, Inclination, True Anomaly,
                        Argument of Perigee, Right Ascension of Ascending Node]
                        if coes=False: [Initial position [x, y, z], Initial velocity [u, v, w]]
        :param tspan: Time span
        :param dt: Time step
        :param target: Target orbit for transfer
        :param coes: Classical orbital elements or simply r0 and v0
        :param deg: True if state input uses degrees instead of radians
        :param mass0: Initial mass of the satellite
        :param cb: Centre body data, if None then Earth will be used
        :param perts: Perturbation effects, see AstroUtils.init_perts
        """
        if perts is None:
            perts = self.init_perts()
        if cb is None:
            cb = DataHandling().import_centre_body('earth')
        if coes:
            self.r0, self.v0 = self.coes2rv(state0, deg=deg, cb=cb)
        else:
            self.r0 = state0[:3]
            self.v0 = state0[3:]

        # Propagation conditions
        self.tspan = tspan
        self.dt = dt
        self.n_steps = int(np.ceil(self.tspan / self.dt))

        # Initialise arrays, memory allocation
        self.ys = np.zeros((self.n_steps, 7))
        self.ts = np.zeros((self.n_steps, 1))

        # COEs only allocated if calculate_coes is used, otherwise it is a waste of memory
        self.coes = None

        # Initial conditions
        self.mass0 = mass0
        self.y0 = self.r0.tolist() + self.v0.tolist() + [self.mass0]
        self.ys[0, :] = self.y0
        self.step = 1

        # Allocate memory for arrays, including initial state 0
        self.rs = self.ys[:, :3]
        self.vs = self.ys[:, 3:6]
        self.masses = self.ys[:, -1]

        # Miscellaneous data
        self.cb = cb
        self.perts = perts
        self.target = target

        self.propagate_orbit()

    def propagate_orbit(self):
        """
        Use scipy ode solver to propogate orbit through integration of position and velocity over time.
        :return: Updates orbit instance's r values over time steps
        """
        # Initiate solver
        solver = ode(self.diffy_q)
        solver.set_integrator('lsoda')
        solver.set_initial_value(self.y0, 0)
        r = self.ys[0, :3]
        r_norm = np.linalg.norm(r)

        if self.target is None:
            target = 400000.0
        else:
            target = self.target

        # Propagate orbit
        while solver.successful() and self.step < self.n_steps and r_norm < target:
            solver.integrate(solver.t + self.dt)
            self.ts[self.step] = solver.t
            self.ys[self.step] = solver.y

            # Check crash condition
            r = self.ys[self.step, :3]
            r_norm = np.linalg.norm(r)
            if r_norm <= self.cb['radius']:
                warnings.warn(f"Object crashed after t = {solver.t} s, at r = {r_norm}", UserWarning)
            self.step += 1

        self.ts = self.ts[:self.step]
        self.rs = self.ys[:self.step, :3]
        self.vs = self.ys[:self.step, 3:6]
        self.masses = self.ys[:self.step, -1]

        # If target orbit is reached before end of time span, set new number of steps
        self.n_steps = self.step

        return

    def diffy_q(self, t, y):
        """
        Differential equation defining the orbit propagation.
        :param t: Current time (used in ode solver)
        :param y: Current state space
        :return: New velocities and accelerations
        """
        # Unpack state space
        rx, ry, rz, vx, vy, vz, mass = y
        dmdt = 0.0
        r = np.array([rx, ry, rz])
        v = np.array([vx, vy, vz])

        # Norm of the radius vector
        r_norm = np.linalg.norm(r)

        # Two body acceleration
        a = -r * self.cb['mu'] / r_norm ** 3

        if self.perts['J2']:
            z2 = r[2] ** 2
            r2 = r_norm ** 2
            tx = r[0] / r_norm * (5 * z2 / r2 - 1)
            ty = r[1] / r_norm * (5 * z2 / r2 - 1)
            tz = r[2] / r_norm * (5 * z2 / r2 - 3)

            a_j2 = 1.5 * self.cb['J2'] * self.cb['mu'] * self.cb['radius'] ** 2 / r_norm ** 4 * np.array([tx, ty, tz])

            a += a_j2

        if self.target:
            # Thrust vector
            a_thrust = (v / np.linalg.norm(v)) * self.perts['thrust'] / mass / 1000.0  # km / s **2

            # Derivative of total mass
            g = 9.81
            dmdt = - np.abs(self.perts['thrust']) / self.perts['isp'] / g
            a += a_thrust

        return [vx, vy, vz, a[0], a[1], a[2], dmdt]

    def calculate_coes(self):
        """
        :return: Updates COEs array using rv2coes util function with existing rs and vs from propagation.
        """
        print('Calculating Classical Orbital Elements (COEs)...')

        self.coes = np.zeros((self.n_steps, 6))

        for n in range(self.n_steps):
            self.coes[n, :] = self.rv2coes(self.rs[n, :], self.vs[n, :], mu=self.cb['mu'], deg=True,
                                           print_results=False)

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
        # axs[0, 2].set_ylabel('Angle (degrees)')

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

    def plot_3d(self, show_plot=False, save_plot=False, title='Test Title'):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        ax.plot(self.rs[:, 0], self.rs[:, 1], self.rs[:, 2], 'b', label='Trajectory')
        ax.plot([self.rs[0, 0]], [self.rs[0, 1]], [self.rs[0, 2]], 'wo', label='Initial Position')

        # Plot central body
        _u, _v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        _x = self.cb['radius'] * np.cos(_u) * np.sin(_v)
        _y = self.cb['radius'] * np.sin(_u) * np.sin(_v)
        _z = self.cb['radius'] * np.cos(_v)
        ax.plot_wireframe(_x, _y, _z, color='lightskyblue', linewidth=0.5)

        # Plot the x,y,z vectors
        l = self.cb['radius'] * 2
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]

        ax.quiver(x, y, z, u, v, w, color='r')

        max_val = np.max(np.abs(self.rs))

        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')

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
            DataHandling().save_figure(fig, title.replace(' ', '_'))
        return


if __name__ == "__main__":
    # Choose orbital body to get data from
    data = cb_data['earth']

    # COEs input
    # Altitude, Eccentricity, Inclination, True anomaly, Argument of Perigee, Right Acsension of Ascending Node

    ikaros = au.make_sat([data['radius'] + 1750, 0.0, 0.0, 0.0, 0.0, 0.0],
                         'IKAROS',
                         10000.0,
                         au.init_perts(J2=True, isp=5000, thrust=5.0),
                         35786.0, )

    iss = au.make_sat([data['radius'] + 418, 0.0000933, 51.6444, 0.0, 193.3222, 77.2969],
                      'ISS',
                      419000.0,
                      au.init_perts(J2=True, isp=4300, thrust=0.327),
                      None)

    crash = au.make_sat([data['radius'] + 400, 0.1, 51.6444, 0.0, 193.3222, 77.2969],
                        'Crash',
                        1000.0,
                        au.init_perts(J2=True, isp=4300, thrust=0.327),
                        None)

    # Create instances
    ikaros0 = OrbitPropagator(state0=ikaros['state'],
                              tspan=3600 * 24 * 365,
                              dt=10.0,
                              target=ikaros['target'],
                              coes=True,
                              deg=True,
                              mass0=ikaros['mass'],
                              cb=data,
                              perts=ikaros['perts'])

    iss0 = OrbitPropagator(state0=iss['state'],
                           tspan=3600 * 24 * 2.0,
                           dt=10.0,
                           target=iss['target'],
                           coes=True,
                           deg=True,
                           mass0=iss['mass'],
                           cb=data,
                           perts=iss['perts'])

    # crash0 = OrbitPropagator(state0=crash['state'],
    #                          tspan=3600 * 24,
    #                          dt=10.0,
    #                          target=crash['target'],
    #                          coes=True,
    #                          deg=True,
    #                          mass0=crash['mass'],
    #                          cb=data,
    #                          perts=crash['perts'])
    #
    # crash0.plot_3d(show_plot=True)

    ikaros0.plot_3d(show_plot=False)

    iss0.plot_3d(show_plot=False, title="ISS Orbit with J2 Perturbation")

    au.plot_n_orbits([ikaros0.rs, iss0.rs],
                     labels=[
                         f"{ikaros['name']} Trajectory {au.get_orbit_time(ikaros0.ts, [0, 0, 1])} days",
                         f"{iss['name']} Trajectory {au.get_orbit_time(iss0.ts)} hrs"],
                     show_plot=True,
                     save_plot=False,
                     title=f"{ikaros['name']} Transfer and {iss['name']} trajectory"
                           f"\nLongest Trajectory {max(au.get_orbit_time(ikaros0.ts, [0, 0, 1]), au.get_orbit_time(iss0.ts, [0, 0, 1]))} days")

    # TODO: Fix/Verify coes calculation, rv2coes and coes2rv
    # ikaros0.calculate_coes()
    # ikaros0.plot_coes(show_plot=True, hours=True)
