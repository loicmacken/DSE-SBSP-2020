import numpy as np


class parabola:
    def __init__(self, radius, depth, up=1):
        self.d = depth  # Dish depth
        self.r = radius  # Dish radius
        self.FP = (self.r ** 2) / (4 * self.d)  # Height focal point from bottom dish

        # Shape
        self.X = np.arange(-radius, radius, 0.5)
        self.Y = (-1) ** up * self.d * (1 - self.X ** 2 / self.r ** 2)

        self.length = np.sqrt(self.r ** 2 + 4 * self.d ** 2) \
                      + self.r ** 2 * np.arcsinh(2 * self.d / self.r) / (2 * self.d)
        self.A = np.pi * self.r / (6 * self.d ** 2) * ((self.r ** 2 + 4 * self.d ** 2) ** (3 / 2) - self.r ** 3)


class truss:
    def __init__(self, x0, x1, y0, y1):
        self.length = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        # Shape
        self.X = np.linspace(x0, x1, 20)
        self.Y = (y1 - y0) / (x1 - x0) * (self.X - x1) + y1


def arrange_relay(relay_offset, r_big, d_big, r_beam, offset, margin):
    gamma = abs(np.arctan((r_big + relay_offset - r_beam) / offset))
    rho = np.pi / 2 + abs(np.arctan(d_big / (r_big + relay_offset))) + np.radians(margin)

    return gamma, rho, margin


def arrange_sting(relay_offset, r_big, r_beam, margin, gamma, rho, beta_marg):
    beta = gamma - np.radians(beta_marg)
    sting_offset = r_big / abs(np.tan(beta))
    alpha = abs(np.arctan((r_big + relay_offset) / sting_offset))
    delta = alpha - beta - np.radians(margin)

    alpha_max = max((np.pi - delta) / 2, (np.pi - rho + alpha) / 2)

    r_sting = r_beam / np.cos(rho / 2)
    A_sting = np.pi * r_sting ** 2

    r_relay = r_beam / np.cos(alpha_max)
    A_relay = np.pi * r_relay ** 2

    return sting_offset, relay_offset, r_sting, r_relay, A_sting, A_relay