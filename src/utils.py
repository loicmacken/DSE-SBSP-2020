from pathlib import Path
import numpy as np
from numpy import ndarray
import math as m
import matplotlib.pyplot as plt
import csv
import PIL
from mpl_toolkits.mplot3d import Axes3D


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_data_root() -> Path:
    return Path(__file__).parent.parent / 'data'


class DataHandling:
    """
    Class for all data handling methods.
    """

    def __init__(self, data_path=None):
        # If no path is given, then the default path will be used
        if data_path is None:
            self.data_path = str(get_data_root())
        else:
            self.data_path = data_path

    def import_centre_body(self, body=None):
        """
        Import centre body data from data/planetary_data.csv
        :param body: Which centre body you want data for
        :return: Either all planetary data in dictionary format, or centre body data as specified
        """
        filename = self.data_path + '/planetary_data.csv'
        planetary_data = dict()
        with open(filename) as csvfile:
            reader = csv.DictReader(csvfile, skipinitialspace=True)
            for row in reader:
                planetary_data[row['name'].lower()] = {'name': row['name'],
                                                       'mass': float(row['mass']),
                                                       'mu': float(row['mu']),
                                                       'radius': float(row['radius']),
                                                       'J2': float(row['J2'])}

        if body is not None:
            body = body.lower()
            if body not in planetary_data.keys():
                raise ValueError("Celestial body not found in data.")
            else:
                return planetary_data[body]
        else:
            return planetary_data

    def save_figure(self, fig, filename, filetype='jpg', path=None):
        """
        Save figure to data/figures (unless different path is given).
        :param fig: Matplotlib figure
        :param filename: Name of file to be saved (without /)
        :param filetype: Image type to save as, jpg or png
        :param path: Directory to save to inside self.data_path, use path='' to save in root data directory.
        :return:
        """
        filename = filename.split("\n")[0]
        if path is None:
            save_path = self.data_path + f'/figures/{filename}.{filetype}'
        else:
            save_path = self.data_path + f'/{path}/{filename}.{filetype}'

        if filetype == 'png':
            fig.savefig(save_path, dpi=300)
        else:
            fig.savefig(save_path)
        return


class AstroUtils:
    """
    Some useful astrodynamics tools.
    """

    @staticmethod
    def d2r(deg):
        """
        Convert degrees to radians.
        :param deg: Value in degrees
        :return: Value in radians
        """
        if deg < 0.0 or deg > 360.0:
            raise ValueError("All degree values should be between 0 and 360.")
        return deg * np.pi / 180.0

    @staticmethod
    def r2d(rad):
        """
        Convert degrees to radians.
        :param rad: Value in radians
        :return: Value in degrees
        """
        if rad < 0.0 or rad > 2 * np.pi:
            raise ValueError("All radian values should be between 0 and 2*pi.")
        return rad * 180.0 / np.pi

    @staticmethod
    def init_perts(J2=False, aero=False, tb=None, thrust=0, isp=0):
        if tb is None:
            tb = []
        return {
            'J2': J2,
            'aero': aero,
            'third_bodies': tb,
            'thrust': thrust,
            'isp': isp
        }

    @staticmethod
    def make_sat(state: list, name: str, mass: float, perts: dict, target=None):
        """
        Create dictionary defining satellite object.
        :param state: list of either initial COEs or r0, v0
        :param name: name of object, used in labeling
        :param target: target orbit (if any)
        :param mass: initial mass
        :param perts: perturbation conditions
        :return: dictionary object containing satellite info
        """
        return {'state': np.array(state),
                'name': name,
                'target': target,
                'mass': mass,
                'perts': perts
                }

    @staticmethod
    def plot_n_orbits(rs: list, labels: list, cb=None, show_plot=False, save_plot=False, title='Test Title'):
        """
        Plot orbits around a centre body sphere.
        :param rs: List of r values for each orbit; i.e. [orbit1.rs, orbit2.rs]
        :param labels: Strings of labels for the plot legend mapped to the rs list
        :param cb: Centre body data, if left as None then Earth will be used
        :param show_plot: Show plot in window if true
        :param save_plot: Save plot to data/figures with title as filename
        :param title: Title of the plot
        :return:
        """

        for i in rs:
            if np.shape(i)[1] != 3:
                raise AttributeError("r values are not three dimensional.")

        if cb is None:
            cb = DataHandling().import_centre_body('earth')
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        n = 0
        for r in rs:
            ax.plot(r[:, 0], r[:, 1], r[:, 2], label=labels[n])
            ax.plot([r[0, 0]], [r[0, 1]], [r[0, 2]], 'o', label=labels[n].split(' ')[0] + ' Initial Position')
            n += 1

        # Plot central body
        _u, _v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        _x = cb['radius'] * np.cos(_u) * np.sin(_v)
        _y = cb['radius'] * np.sin(_u) * np.sin(_v)
        _z = cb['radius'] * np.cos(_v)
        ax.plot_wireframe(_x, _y, _z, color='lightskyblue', linewidth=0.5)

        # Plot the x,y,z vectors
        l = cb['radius'] * 2
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]

        ax.quiver(x, y, z, u, v, w, color='r')

        max_val = 0
        for r in rs:
            max_val = max(max_val, np.max(np.abs(r)))

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
        if save_plot:
            DataHandling().save_figure(fig, title.replace(' ', '_'))
        return

    def coes2rv(self, state: list, deg=False, cb=None):
        """
        Classical Orbital Elements conversion to r and v vectors.
        :param state: Altitude [km], Eccentricity, Inclination, True Anomaly, Argument of Perigee,
        Right Ascension of Ascending Node
        :param deg: Input in degrees or radians
        :param cb: Central body data
        :return: r and v in inertial frame
        """
        if cb is None:
            cb = DataHandling().import_centre_body('Earth')
            mu = cb['mu']
        else:
            mu = cb['mu']

        a, e, i, ta, aop, raan = state

        # Ensure all angles are in radians
        if deg:
            i = self.d2r(i)
            ta = self.d2r(ta)
            aop = self.d2r(aop)
            raan = self.d2r(raan)

        E = self.ecc_anomaly([ta, e], 'tae')

        r_norm = a * (1 - e ** 2) / (1 + e * m.cos(ta))

        # Calculate r and v vectors in perifocal frame
        r_perif = r_norm * np.array([m.cos(ta), m.sin(ta), 0])
        v_perif = m.sqrt(mu * a) / r_norm * np.array([-m.sin(E), m.cos(E) * m.sqrt(1 - e ** 2), 0])

        # Rotation matrix from perifocal to ECI
        perif2eci = np.transpose(self.eci2perif(raan, aop, i))

        # Calculate r and v vectors in inertial frame
        r = np.dot(perif2eci, r_perif)
        v = np.dot(perif2eci, v_perif)

        return r, v

    def rv2coes(self, r, v, mu=None, deg=True, print_results=False):
        if mu is None:
            cb = DataHandling().import_centre_body('Earth')
            mu = cb['mu']
        # Norm of position vector
        r_norm = np.linalg.norm(r)

        # Specific angular momentum
        h = np.cross(r, v)
        h_norm = np.linalg.norm(h)

        # Inclination
        i = m.acos(h[2] / h_norm)

        # Eccentricity vector
        # e = ((np.linalg.norm(v) ** 2 - mu / r_norm) * r - np.dot(r, v) * v) / mu
        e = (np.cross(v, h) / mu) - (r / r_norm)
        e_norm = np.linalg.norm(e)

        # Node line
        N = np.cross([0, 0, 1], h)
        N_norm = np.linalg.norm(N)

        # RAAN
        raan = m.acos(N[0] / N_norm) if not np.nan else 0.0
        if N[1] < 0:
            raan = 2 * np.pi - raan

        # Argument of Perigee
        aop = m.acos(np.dot(N, e) / N_norm / e_norm) if not np.nan else 0.0
        if e[2] < 0:
            aop = 2 * np.pi - aop

        # True anomaly
        ta = m.acos(round(np.dot(e, r) / e_norm / r_norm, 10))
        if np.dot(r, v) < 0:
            ta = 2 * np.pi - ta

        # semi-major axis
        a = r_norm * (1 + e_norm * m.cos(ta)) / (1 - e_norm ** 2)

        if print_results:
            print(f'Semi-major axis [km]: {a}')
            print(f'Eccentricity: {e_norm}')
            print(f'Inclination [deg]: {self.r2d(i)}')
            print(f'Right Ascension of Ascending Node [deg]: {self.r2d(raan)}')
            print(f'Argument of Perigee [deg]: {self.r2d(aop)}')
            print(f'True Anomaly [deg]: {self.r2d(ta)}')
            print()

        # Convert to degrees if specified
        if deg:
            return [a, e_norm, self.r2d(i), self.r2d(ta), self.r2d(aop), self.r2d(raan)]
        else:
            return [a, e_norm, i, ta, aop, raan]

    @staticmethod
    def eci2perif(raan, aop, i):
        """
        Convert Earth-centred Inertial to Perifocal rotation matrix. All angles in radians.
        :param raan: Right Ascension of Ascending Node
        :param aop: Argument of Perigee
        :param i: Inclination
        :return: raan, aop, i in perifocal coordinate system; three axes.
        """
        row0 = [-m.sin(raan) * m.cos(i) * m.sin(aop) + m.cos(raan) * m.cos(aop),
                m.cos(raan) * m.cos(i) * m.sin(aop) + m.sin(raan) * m.cos(aop),
                m.sin(i) * m.sin(aop)]
        row1 = [-m.sin(raan) * m.cos(i) * m.cos(aop) - m.cos(raan) * m.sin(aop),
                m.cos(raan) * m.cos(i) * m.cos(aop) - m.sin(raan) * m.sin(aop),
                m.sin(i) * m.cos(aop)]
        row2 = [m.sin(raan) * m.sin(i),
                -m.cos(raan) * m.sin(i),
                m.cos(i)]
        return np.array([row0, row1, row2])

    @staticmethod
    def ecc_anomaly(arr, method, tol=1e-8):
        """
        Iteratively determine the eccentricity anomaly.
        :param arr: [Mean eccentricity, eccentricity]
        :param method: Type of method to be used; newton or trial and error
        :param tol: Comparison tolerance.
        :return: False if algorithm did not converge, else return eccentricity anomaly
        """
        E1 = None
        if method == 'newton':
            # Newton's method for iteratively finding E
            Me, e = arr
            if Me < np.pi / 2.0:
                E0 = Me + e / 2.0
            else:
                E0 = Me - e
            # Arbitrary number of steps in range()
            for n in range(200):
                ratio = (E0 - e * np.sin(E0) - Me) / (1 - e * np.cos(E0))
                if abs(ratio) < tol:
                    return E0 if n == 0 else E1
                else:
                    E1 = E0 - ratio
                    E0 = E1
            # Did not converge
            return False
        elif method == 'tae':
            ta, e = arr
            return 2 * m.atan(m.sqrt((1 - e) / (1 + e)) * m.tan(ta / 2.0))
        else:
            print('Invalid method for eccentric anomaly.')

    @staticmethod
    def get_orbit_time(ts: ndarray, units=None):
        """

        :param ts: Orbit Propagation times as (nsteps, 1) numpy array
        :param units: [s, hrs, days], default is hrs
        :return: time in given units
        """
        if units is None:
            units = [0, 1, 0]
        if sum(units) > 1:
            raise ValueError("Only one unit can be given")

        if units == [1, 0, 0]:
            t = round(ts[-1][0], 5)
        elif units == [0, 1, 0]:
            t = round(ts[-1][0] / 3600.0, 4)
        elif units == [0, 0, 1]:
            t = round(ts[-1][0] / 3600.0 / 24.0, 3)
        else:
            raise ValueError("Units must be one hot encoded.")

        return t


if __name__ == "__main__":
    proot = get_project_root()
    droot = get_data_root()
