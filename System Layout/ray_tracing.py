#this program can be used to trace the rays and see how they diverge
#for some reason the step size, aka the second to last argument in the init needs to be smaller than or equal to 0.01 
# i assume it's because otherwise it chooses the same point instead of the point one step over


import numpy as np
import matplotlib.pyplot as plt
from divergence import divergence_angle

class Raytrace:
    def __init__(self, ray_amount, parabola1_radius, parabola2_radius, parabola1_depth ,step , up=1):
        self.ray_amount = ray_amount
        self.parabola1_radius = parabola1_radius
        self.parabola2_radius = parabola2_radius
        self.parabola1_depth = parabola1_depth
        self.step = step
        self.parabola1_angle = 0
        self.up = up
        self.FP = (self.parabola1_radius ** 2) / (4 * self.parabola1_depth)  # Height focal point from bottom dish
        self.FP_coord = [0,self.FP - self.parabola1_depth]
        self.divergence = divergence_angle
        self.X = np.arange(-self.parabola1_radius, self.parabola1_radius + self.step, self.step)
        self.Y = (-1) ** self.up * self.parabola1_depth * (1 - self.X ** 2 / self.parabola1_radius ** 2)

        self.length = np.sqrt(self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) \
                      + self.parabola1_radius ** 2 * np.arcsinh(2 * self.parabola1_depth / self.parabola1_radius) / (2 * self.parabola1_depth)
        self.A = np.pi * self.parabola1_radius / (6 * self.parabola1_depth ** 2) * ((self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) ** (3 / 2) - self.parabola1_radius ** 3)

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

    def ray_angle(self):
        return()
   
    def reflection(self,starting_location_x): #first calculate the intersect point if beam straight, then calculate intersect point if divergent, then calculate angle towards focal point
        self.starting_location_x = starting_location_x #the x location where the ray begins
        self.intersect_unangled_index = min(range(len(self.X)), key = lambda i: abs(self.X[i]-self.starting_location_x)) #the index of the number in the list of parabola x values that is closest to the start x location of the ray
        self.intersect_unangled_local_gradient =  (np.pi / 4) - (np.arctan( (self.Y[self.intersect_unangled_index] \
                                                    - self.Y[self.intersect_unangled_index - 1]) / (self.X[self.intersect_unangled_index] - self.X[self.intersect_unangled_index - 1]) )) #the local gradient of the parabola between the closest x and the previous x value
        self.intersect_unangled_distance_y = abs(self.Y[self.intersect_unangled_index])
        self.intersect_angled_x = self.starting_location_x - np.tan(self.divergence + self.parabola1_angle) * self.intersect_unangled_distance_y \
                                     * (np.tan(self.intersect_unangled_local_gradient) / (np.tan(self.intersect_unangled_local_gradient) + np.tan(self.divergence + self.parabola1_angle)) ) #calculate the x coordinate of the angled beam through some geometry calculations-> dist=wtana/(tana+tanb)
        self.intersect_angled_index = min(range(len(self.X)), key = lambda i: abs(self.X[i]-self.intersect_angled_x))  #same story as other index but now for angled intersect
        self.intersect_angled_XY =  [self.X[self.intersect_angled_index], self.Y[self.intersect_angled_index] ] #xy coordinate of the intersect point if the beam is divergent
        self.intersect_angled_gradient = (np.arctan( (self.Y[self.intersect_angled_index] \
                                                    - self.Y[self.intersect_angled_index - 1]) / (self.X[self.intersect_angled_index] - self.X[self.intersect_angled_index - 1]) )) #same thing as unangled gradient
        self.intersect_angled_angle = self.divergence + self.parabola1_angle + 2 * (self.intersect_angled_gradient - (self.divergence + self.parabola1_angle)) #angle of beam wrt horizontal
        return(self.intersect_angled_XY,self.intersect_angled_angle)
    
    def focal_abberation(self):
        






x = Raytrace(20,434,10,434/3,0.01)
print(x.reflection(400))



"""
xval = x.parabola()[0]
yval = x.parabola()[1]
plt.plot(xval,yval)
plt.axis('equal')
plt.show()"""