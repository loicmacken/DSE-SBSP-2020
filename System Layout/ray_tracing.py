import numpy as np
import matplotlib.pyplot as plt
#define parabola here, maybe import from parabola.py?
"""pv area = bus power
aperture lens
intake area
"""

#set amount of rays
ray_amount = 1

class Raytrace:

    def __init__(self,ray_amount,parabola1_radius,parabola2_radius, parabola1_depth, up=1):
        self.ray_amount = ray_amount
        self.parabola1_radius = parabola1_radius
        self.parabola2_radius = parabola2_radius
        self.parabola1_depth = parabola1_depth
        self.FP = (self.parabola1_radius ** 2) / (4 * self.parabola1_depth)  # Height focal point from bottom dish
        self.up = up

    def parabola(self): 
        self.X = np.arange(-self.parabola1_radius, self.parabola1_radius+0.5, 0.5)
        self.Y = (-1) ** self.up * self.parabola1_depth * (1 - self.X ** 2 / self.parabola1_radius ** 2)

        self.length = np.sqrt(self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) \
                      + self.parabola1_radius ** 2 * np.arcsinh(2 * self.parabola1_depth / self.parabola1_radius) / (2 * self.parabola1_depth)
        self.A = np.pi * self.parabola1_radius / (6 * self.parabola1_depth ** 2) * ((self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) ** (3 / 2) - self.parabola1_radius ** 3)
        return(self.X,self.Y)

    def ray_start(self):
        self.intake_radius = self.parabola1_radius - self.parabola2_radius
        self.intake_left = self.intake_radius / 2
        self.intake_right= self.intake_radius / 2

        if (self.ray_amount % 2) == 0:
            self.rays_left = self.ray_amount / 2
            self.rays_right = self.ray_amount / 2
        else:
            self.rays_left = (self.ray_amount + 1) / 2
            self.rays_right = (self.ray_amount - 1) / 2
        self.step_size_left = self.intake_left / (self.rays_left + 1)
        self.step_size_right = self.intake_right / (self.rays_right + 1)
        self.starting_locations_left = []
        self.starting_locations_right = []

        for self.i in np.arange(0 + self.step_size_left, self.intake_left, self.step_size_left):
            self.starting_locations_left.append(self.i)

        for self.j in np.arange(self.parabola1_radius - self.step_size_right, self.parabola1_radius-self.intake_right, -self.step_size_right):
            self.starting_locations_right.append(self.j)

        return(self.starting_locations_left, self.starting_locations_right)

        






x = Raytrace(20,434,10,434/3)

"""
xval = x.parabola()[0]
yval = x.parabola()[1]
plt.plot(xval,yval)
plt.axis('equal')
plt.show()"""