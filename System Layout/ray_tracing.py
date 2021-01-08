#this program can be used to trace the rays and see how they diverge
#for some reason the step size, aka the second to last argument in the init needs to be smaller than or equal to 0.01 
# i assume it's because otherwise it chooses the same point instead of the point one step over


import numpy as np
import matplotlib.pyplot as plt
from divergence import divergence_angle
class Parabolas:
    def __init__(self, parabola1_radius, parabola2_radius, parabola1_depth, parabola2_offset,step):
        self.parabola1_radius = parabola1_radius
        self.parabola2_radius = parabola2_radius
        self.parabola1_depth = parabola1_depth
        self.parabola2_depth = self.parabola2_radius * (self.parabola1_depth / self.parabola1_radius)
        self.parabola2_offset = parabola2_offset
        self.step = step

    def parab_big(self):
        self.X1 = np.arange(-self.parabola1_radius, self.parabola1_radius + self.step, self.step)
        self.Y1 = (-1) * self.parabola1_depth * (1 - self.X1 ** 2 / self.parabola1_radius ** 2)
        """self.length = np.sqrt(self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) \
                      + self.parabola1_radius ** 2 * np.arcsinh(2 * self.parabola1_depth / self.parabola1_radius) / (2 * self.parabola1_depth)
        self.A = np.pi * self.parabola1_radius / (6 * self.parabola1_depth ** 2) * ((self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) ** (3 / 2) - self.parabola1_radius ** 3)"""
        return(self.X1, self.Y1)
    
    def parab_small(self):
        self.X2 = np.arange(-self.parabola2_radius, self.parabola2_radius + self.step, self.step)
        self.Y2 = self.parabola2_depth * (1 - self.X2 ** 2 / self.parabola2_radius ** 2)
        self.Y2 = self.Y2 + self.parabola2_offset
        return(self.X2,self.Y2)

class Raytrace:
    def __init__(self, ray_amount, parabola1_radius, parabola2_radius, parabola1_depth,parabola2_offset ,step, x1, y1, x2, y2 , up=1):
        self.ray_amount = ray_amount
        self.parabola1_radius = parabola1_radius
        self.parabola2_radius = parabola2_radius
        self.parabola1_depth = parabola1_depth
        self.parabola2_offset = parabola2_offset
        self.step = step
        self.parabola1_angle = 0
        self.up = up
        self.FP = (self.parabola1_radius ** 2) / (4 * self.parabola1_depth)  # Height focal point from bottom dish
        self.FP_coord = [0, self.FP - self.parabola1_depth]
        self.divergence = divergence_angle + self.parabola1_angle
        self.parabola2_depth = self.parabola2_radius * (self.parabola1_depth / self.parabola1_radius)
        self.X1 = x1
        self.X2 = x2
        self.Y1 = y1
        self.Y2 = y2

    def ray_start(self):
        self.intake_radius = 2 * self.parabola1_radius - 2 * self.parabola2_radius
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

        for self.i in np.arange(- self.parabola1_radius + self.step_size_left, -self.parabola1_radius + self.intake_left, self.step_size_left):
            self.starting_locations_left.append(self.i)

        for self.j in np.arange(self.parabola1_radius - self.step_size_right, self.parabola1_radius-self.intake_right, -self.step_size_right):
            self.starting_locations_right.append(self.j)

        return(self.starting_locations_left, self.starting_locations_right)

    def ray_angle(self):
        return()
   
    def reflection(self, starting_location_x): #first calculate the intersect point if beam straight, then calculate intersect point if divergent, then calculate angle towards focal point
        self.starting_location_x = starting_location_x #the x location where the ray begins
        self.intersect_unangled_index = min(range(len(self.X1)), key = lambda i: abs(self.X1[i]-self.starting_location_x)) #the index of the number in the list of parabola x values that is closest to the start x location of the ray
        self.intersect_unangled_local_gradient = (np.pi/2) - (np.arctan( (self.Y1[self.intersect_unangled_index - 1] \
                                                    - self.Y1[self.intersect_unangled_index]) / (self.X1[self.intersect_unangled_index] - self.X1[self.intersect_unangled_index - 1]) )) #the local gradient of the parabola between the closest x and the previous x value (angle to horizontal axis)
        self.intersect_unangled_distance_y = abs(self.Y1[self.intersect_unangled_index]) #distance between starting point and unangled intersection point
        self.intersect_angled_y = - ( self.intersect_unangled_distance_y - (self.intersect_unangled_distance_y * np.tan(self.divergence)) / (np.tan(self.divergence) + np.tan(self.intersect_unangled_local_gradient)))
        self.intersect_angled_index = min(range(len(self.Y1)), key = lambda i: abs(self.Y1[i]-self.intersect_angled_y)) #same story as other index but now for angled intersect
        self.intersect_angled_XY =  [self.X1[self.intersect_angled_index], self.Y1[self.intersect_angled_index] ] #xy coordinate of the intersect point if the beam is divergent
        self.intersect_angled_gradient = (np.arctan( (self.Y1[self.intersect_angled_index - 1] \
                                                    - self.Y1[self.intersect_angled_index]) / (self.X1[self.intersect_angled_index] - self.X1[self.intersect_angled_index - 1]) )) #same thing as unangled gradient
        self.intersect_angled_angle = self.divergence + 2 * (self.intersect_angled_gradient - (self.divergence)) #angle of beam wrt vertical I/
        return(self.intersect_angled_XY,self.intersect_angled_angle)

    def focal_abberation(self, starting_location_x):
        self.starting_location_x = starting_location_x
        self.reflect_xy = self.reflection(self.starting_location_x)[0]
        self.reflect_angle = self.reflection(self.starting_location_x)[1]
        self.focal_y = self.FP_coord[1]
        self.abberation_y = self.focal_y 
        self.abberation_x = self.reflect_xy[0] + (self.abberation_y -self.reflect_xy[1]) * np.tan(self.reflect_angle)
        return(self.abberation_x, self.abberation_y)

    def collimation(self, starting_location_x): #find point where equation of beam is equal to equation of parabola
        self.starting_location_x = starting_location_x
        self.a = 1 / np.tan(self.reflection(self.starting_location_x)[1])
        """self.b = self.reflection(starting_location_x)[0][1] / (self.reflection(starting_location_x)[0][0] * self.a)#for ax+b = y b is equal to y/ax where a is tan(90-angle to vertical), y and x are the intersect point#self.reflection(self.starting_location_x)[0][0] + self.a * self.reflection(self.starting_location_x)[0][1]
        """
        self.b = (- self.reflection(starting_location_x)[0][0] * self.a) + self.reflection(starting_location_x)[0][1]
        self.a_quadtratic = self.parabola2_depth / (self.parabola2_radius ** 2)
        self.b_quadtratic = self.a
        self.c_quadtratic = self.b - self.a_quadtratic - self.parabola2_offset
        self.x_sol_pos = (- self.b_quadtratic + np.sqrt((self.b_quadtratic ** 2) - 4 * self.a_quadtratic * self.c_quadtratic)) / (2 * self.a_quadtratic)
        self.x_sol_neg = (- self.b_quadtratic - np.sqrt((self.b_quadtratic ** 2) - 4 * self.a_quadtratic * self.c_quadtratic)) / (2 * self.a_quadtratic)
        self.y_sol_pos = self.Y2[min(range(len(self.X2)), key = lambda i: abs(self.X2[i]-self.x_sol_pos))]
        self.y_sol_neg = self.Y2[min(range(len(self.X2)), key = lambda i: abs(self.X2[i]-self.x_sol_neg))] 
        if -self.parabola2_radius <= self.x_sol_pos <= self.parabola2_radius:
            self.x_sol = self.x_sol_pos
            self.y_sol = self.y_sol_pos
        else:
            self.x_sol = self.x_sol_neg
            self.y_sol = self.y_sol_neg
        #find intersection with b2-4ac
        self.collimation_xy = [self.x_sol, self.y_sol]
        return(self.collimation_xy)




y = Parabolas(400,25,71,622.81, 0.001)
x1,y1 = y.parab_big()
x2,y2 = y.parab_small()
x = Raytrace(10,400,25,71,622.81,0.001,x1,y1,x2,y2)
print(x.collimation(-100))
line = (-100,-100,x.focal_abberation(-100)[0],x.collimation(-100)[0])
liney= (0,x.reflection(-100)[0][1],x.focal_abberation(-100)[1],x.collimation(-100)[1])
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(line,liney)
plt.axis('equal')
plt.show()