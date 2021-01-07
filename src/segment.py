import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math as m
from scipy import integrate


class Parabola:

    def __init__(self, radius, depth, start, n_steps):
        self.radius = radius
        self.depth = depth
        self.n_steps = n_steps
        self.x = np.linspace(start, self.radius, n_steps)
        self.y = np.zeros(len(self.x))
        self.y[0] = self.func(self.x[0], self.depth, self.radius)

        self.step = 1
        self.discretize()

    def func(self, x, depth, radius):
        return depth * (1 - x ** 2 / radius ** 2)

    def discretize(self):
        for seg in range(self.n_steps):
            self.y[seg] = self.depth * (1 - self.x[seg] ** 2 / self.radius ** 2)

    def interpcurve(self):

        # equally spaced in arclength
        N = np.transpose(np.linspace(0, 1, self.n_steps))

        # how many points will be uniformly interpolated?
        nt = N.size

        # number of points on the curve
        n = self.x.size
        pxy = np.array((self.x, self.y)).T
        p1 = pxy[0, :]
        pend = pxy[-1, :]
        last_segment = np.linalg.norm(np.subtract(p1, pend))
        epsilon = 10 * np.finfo(float).eps

        # IF the two end points are not close enough lets close the curve
        if last_segment > epsilon * np.linalg.norm(np.amax(abs(pxy), axis=0)):
            pxy = np.vstack((pxy, p1))
            nt = nt + 1
        else:
            print('Contour already closed')

        pt = np.zeros((nt, 2))

        # Compute the chordal arclength of each segment.
        chordlen = (np.sum(np.diff(pxy, axis=0) ** 2, axis=1)) ** (1 / 2)
        # Normalize the arclengths to a unit total
        chordlen = chordlen / np.sum(chordlen)
        # cumulative arclength
        cumarc = np.append(0, np.cumsum(chordlen))

        tbins = np.digitize(N, cumarc)  # bin index in which each N is in

        # catch any problems at the ends
        tbins[np.where(tbins <= 0 | (N <= 0))] = 1
        tbins[np.where(tbins >= n | (N >= 1))] = n - 1

        s = np.divide((N - cumarc[tbins]), chordlen[tbins - 1])
        pt = pxy[tbins, :] + np.multiply((pxy[tbins, :] - pxy[tbins - 1, :]), (np.vstack([s] * 2)).T)

        return pt


if __name__ == '__main__':
    p = Parabola(434.0, 71.32, 13.35, 40)

    pt = p.interpcurve()
    x = np.zeros(len(pt))
    y = np.zeros(len(pt))
    for i, point in enumerate(pt):
        x[i] = point[0]
        y[i] = point[1]

    first_seg = np.linalg.norm(np.array(x[1], y[1])) - np.linalg.norm(np.array(x[0], y[0]))
    last_seg = np.linalg.norm(np.array(x[-2], y[-2])) - np.linalg.norm(np.array(x[-3], y[-3]))
    print(first_seg)
    print(last_seg)
    plt.plot(x, y, 'bo')

    plt.show()
