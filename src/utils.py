from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_data_root() -> Path:
    return Path(__file__).parent.parent / 'data'


def import_data(body):
    filename = str(get_data_root()) + '/planetary_data.csv'
    planetary_data = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            planetary_data[row['name'].lower()] = {'name': row['name'],
                                                   'mass': float(row['mass']),
                                                   'mu': float(row['mu']),
                                                   'radius': float(row['radius'])}
    if body not in planetary_data.keys():
        raise ValueError("Celestial body not found in data.")
    return planetary_data[body]


def plot_n_orbits(rs, labels, cb=None, show_plot=False, save_plot=False, title='Test Title'):
    if cb is None:
        cb = import_data('earth')
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    n = 0
    for r in rs:
        ax.plot(r[:, 0], r[:, 1], r[:, 2], label=labels[n])
        ax.plot([r[0, 0]], [r[0, 1]], [r[0, 2]], 'bo')
        n += 1

    # Plot central body
    _u, _v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    _x = cb['radius'] * np.cos(_u) * np.sin(_v)
    _y = cb['radius'] * np.sin(_u) * np.sin(_v)
    _z = cb['radius'] * np.cos(_v)
    ax.plot_surface(_x, _y, _z, cmap='Greens')

    # Plot the x,y,z vectors
    l = cb['radius'] * 2
    x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]

    ax.quiver(x, y, z, u, v, w, color='r')

    max_val = np.max(np.abs(rs))

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
        save_path = str(get_data_root()) + '/figures'
        fig.savefig(save_path + f"/{title}")
    return


if __name__ == "__main__":
    proot = get_project_root()
    droot = get_data_root()
