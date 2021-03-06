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
        self.angles = np.zeros(len(self.x) - 1)
        self.y[0] = self.func(self.x[0])

        self.step = 1
        self.discretize()

    def func(self, x):
        return self.depth * (1 - x ** 2 / self.radius ** 2)

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

        self.x = np.zeros(len(pt))
        self.y = np.zeros(len(pt))

        for i, point in enumerate(pt):
            self.x[i] = point[0]
            self.y[i] = point[1]

        return pt

    def segment_curve(self):
        for i, x in enumerate(self.x):
            self.y[i] = self.func(x)

    def calc_angles(self):
        points = list(zip(self.x, self.y))
        for i in range(len(points) - 1):
            # seg1 = np.linalg.norm(points[i + 1]) - np.linalg.norm(points[i])
            seg1x = points[i + 1][0] - points[i][0]
            seg1y = points[i + 1][1] - points[i][1]
            seg1 = m.sqrt(seg1x ** 2 + seg1y ** 2)
            angle = m.acos(seg1x / seg1)

            #
            # unit1 = points[i] / np.linalg.norm(points[i])
            # unit2 = points[i + 1] / np.linalg.norm(points[i + 1])
            # angle = np.arccos(np.dot(unit1, unit2))
            self.angles[i] = angle
        return

    def get_segments(self):
        segments = {'coords': [], 'lengths': []}
        points = list(zip(self.x, self.y))
        for i in range(len(points) - 1):
            segments['coords'].append((points[i], points[i + 1]))
            seg1x = points[i + 1][0] - points[i][0]
            seg1y = points[i + 1][1] - points[i][1]
            seg_length = m.sqrt(seg1x ** 2 + seg1y ** 2)
            segments['lengths'].append(seg_length)

        return segments


if __name__ == '__main__':
    pv_width = 3.25
    p = Parabola(482.0, 92.38, 25 + pv_width, 23)

    p.segment_curve()

    # pt = p.interpcurve()

    p.calc_angles()

    segments = p.get_segments()

    # angles = np.cumsum(p.angles)
    # print(len(angles))
    print(len(segments['lengths']))

    radius = 0.0
    i = 0
    for angle in p.angles:
        radius += m.cos(angle) * segments['lengths'][i]
        i += 1

    radius += 25 + pv_width
    print(radius)

    print(p.angles)
    print(segments['lengths'])

    plt.plot(p.x, p.y, 'bo', p.x, p.y, 'k')

    plt.show()
