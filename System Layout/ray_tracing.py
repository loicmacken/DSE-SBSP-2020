#this program can be used to trace the rays and see how they diverge
#for some reason the step size, aka the second to last argument in the init needs to be smaller than or equal to 0.01 
# i assume it's because otherwise it chooses the same point instead of the point one step over


import numpy as np
import matplotlib.pyplot as plt
from divergence import divergence_angle
import datetime
class Parabolas:
    def __init__(self, parabola1_radius, parabola2_radius, parabola1_depth, parabola2_offset,step):
        self.parabola1_radius = parabola1_radius
        self.parabola2_radius = parabola2_radius
        self.parabola1_depth = parabola1_depth
        self.parabola2_depth = self.parabola2_radius * (self.parabola1_depth / self.parabola1_radius)
        self.parabola2_offset = parabola2_offset
        self.pv_width = 14.71
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
    
    def height_aperture(self):
        self.radius_flat = self.parabola2_radius + self.pv_width
        self.x = self.parab_big()[0]
        self.height_index = min(range(len(self.x)), key = lambda i: abs(self.x[i] + self.radius_flat))
        self.height = self.parab_big()[1][self.height_index]
        self.start = self.parab_big()[0][self.height_index]
        return(self.height,self.start)

class Raytrace:
    def __init__(self, ray_amount, parabola1_radius, parabola2_radius, parabola1_depth,parabola2_offset ,aperture_depth, step, x1, y1, x2, y2 , up=1):
        self.ray_amount = ray_amount
        self.parabola1_radius = parabola1_radius
        self.parabola2_radius = parabola2_radius
        self.parabola1_depth = parabola1_depth
        self.parabola2_offset = parabola2_offset
        self.aperture_depth = aperture_depth
        self.step = step
        self.parabola1_angle = 0
        self.up = up
        self.FP = (self.parabola1_radius ** 2) / (4 * self.parabola1_depth)  # Height focal point from bottom dish
        self.FP_coord = [0, self.FP - self.parabola1_depth]
        self.divergence =divergence_angle + self.parabola1_angle
        self.parabola2_depth = self.parabola2_radius * (self.parabola1_depth / self.parabola1_radius)
        self.X1_full = x1
        self.X2_full = x2
        self.Y1_full = y1
        self.Y2_full = y2
        self.pv_width = 14.71
        print("spot size",2 * (np.tan(self.divergence) *35786 * 1000) + self.parabola2_radius * 2)

    def ray_start(self): #add pv here
        self.intake_radius = 2 * self.parabola1_radius - 2 * self.parabola2_radius -2 * self.pv_width
        self.intake_left = self.intake_radius / 2
        self.rays_left = self.ray_amount
        self.step_size_left = self.intake_left / (self.rays_left + 1)
        self.starting_locations_left = []
        for self.i in np.arange(- self.parabola1_radius + self.step_size_left, -self.parabola1_radius + self.intake_left, self.step_size_left):
            self.starting_locations_left.append(self.i)
        return(self.starting_locations_left)

    def reflection(self, starting_location_x, X1_slice, Y1_slice): #first calculate the intersect point if beam straight, then calculate intersect point if divergent, then calculate angle towards focal point
        self.X1 = X1_slice
        self.Y1 = Y1_slice
        self.starting_location_x = starting_location_x #the x location where the ray begins
        self.intersect_unangled_index = min(range(len(self.X1)), key = lambda i: abs(self.X1[i]-self.starting_location_x)) #the index of the number in the list of parabola x values that is closest to the start x location of the ray
        self.intersect_unangled_local_gradient = - 2 * (self.parabola1_depth / (self.parabola1_radius ** 2)) * self.starting_location_x#np.arctan(abs(self.Y1[self.intersect_unangled_index] - self.Y1[self.intersect_unangled_index - 1])  / abs(self.X1[self.intersect_unangled_index] - self.X1[self.intersect_unangled_index - 1]))
        #the local gradient of the parabola between the closest x and the previous x value (angle to horizontal axis)
        self.intersect_unangled_distance_y = abs(self.Y1[self.intersect_unangled_index]) #distance between starting point and unangled intersection point
        self.intersect_angled_dy = (self.intersect_unangled_distance_y * np.tan(self.divergence)) / (np.tan((np.pi / 2) - self.intersect_unangled_local_gradient) + np.tan(self.divergence))
        self.intersect_angled_y = self.Y1[self.intersect_unangled_index] + self.intersect_angled_dy
        self.intersect_angled_index = min(range(len(self.Y1)), key = lambda i: abs(self.Y1[i]-self.intersect_angled_y))
        #self.intersect_angled_dx = self.intersect_angled_dy * np.tan((np.pi / 2) - self.intersect_unangled_local_gradient)
        self.intersect_angled_XY = (self.X1[self.intersect_angled_index], self.Y1[self.intersect_angled_index])
        self.intersect_angled_local_gradient =- 2 * (self.parabola1_depth / (self.parabola1_radius ** 2)) * self.intersect_angled_XY[0]#np.arctan(abs(self.Y1[self.intersect_angled_index] - self.Y1[self.intersect_angled_index - 1])  / abs(self.X1[self.intersect_angled_index] - self.X1[self.intersect_angled_index - 1]))
        self.intersect_angled_angle = 2 * self.intersect_angled_local_gradient - self.divergence #angle of beam wrt vertical I/
        self.check_angle = np.arctan(abs(self.Y1[self.intersect_angled_index] - self.Y1[self.intersect_angled_index - 1])  / abs(self.X1[self.intersect_angled_index] - self.X1[self.intersect_angled_index - 1]))
        print(self.intersect_angled_angle, self.intersect_angled_local_gradient*2, self.intersect_unangled_local_gradient*2, "gradients", self.check_angle-self.intersect_angled_local_gradient)
        return(self.intersect_angled_XY,self.intersect_angled_angle)
# abs(- 2 * (self.parabola1_depth / (self.parabola1_radius ** 2)) * self.intersect_angled_XY[0])abs(- 2 * (self.parabola1_depth / (self.parabola1_radius ** 2)) * self.starting_location_x)
    def focal_abberation(self, starting_location_x, XY_reflection, angle_reflection):
        self.starting_location_x = starting_location_x
        self.reflect_xy = XY_reflection 
        self.reflect_angle = angle_reflection
        self.focal_y = self.FP_coord[1]
        self.abberation_y = self.focal_y 
        self.abberation_x = self.reflect_xy[0] + (self.abberation_y -self.reflect_xy[1]) * np.tan(self.reflect_angle)
        self.a = np.sqrt((self.reflect_xy[0] - self.abberation_x)**2 + (self.reflect_xy[1] - self.abberation_y)**2)
        self.b = np.sqrt((self.reflect_xy[0] - self.FP_coord[0])**2 + (self.reflect_xy[1] - self.FP_coord[1])**2)
        self.c = abs(self.abberation_x - self.FP_coord[0])
        self.abang = np.arccos(((self.a**2) + (self.b**2) - (self.c**2))/(2 * self.a * self.b))
        print("focal abberation", self.abberation_x - self.FP_coord[0])
        #print("reflect to fp",np.sqrt((self.reflect_xy[0] - self.abberation_x)**2 + (self.reflect_xy[1] - self.abberation_y)**2))
        #print("ideal reflect to fp", np.sqrt((self.reflect_xy[0] - self.FP_coord[0])**2 + (self.reflect_xy[1] - self.FP_coord[1])**2))
        #print('angle abberation', self.abang)
        return(self.abberation_x, self.abberation_y)

    def collimation(self, starting_location_x, XY_reflection, angle_reflection ): #find point where equation of beam is equal to equation of parabola
        self.starting_location_x = starting_location_x
        self.reflect_xy = XY_reflection
        self.reflect_angle = angle_reflection
        self.a = 1 / np.tan(self.reflect_angle)
        self.b = (- self.reflect_xy[0] * self.a) + self.reflect_xy[1]
        self.a_quadtratic = self.parabola2_depth / (self.parabola2_radius ** 2)
        self.b_quadtratic = self.a
        self.c_quadtratic = self.b - self.a_quadtratic - self.parabola2_offset
        self.x_sol_pos = (- self.b_quadtratic + np.sqrt((self.b_quadtratic ** 2) - 4 * self.a_quadtratic * self.c_quadtratic)) / (2 * self.a_quadtratic)
        self.x_sol_neg = (- self.b_quadtratic - np.sqrt((self.b_quadtratic ** 2) - 4 * self.a_quadtratic * self.c_quadtratic)) / (2 * self.a_quadtratic)
        self.x_index_pos = min(range(len(self.X2_full)), key = lambda i: abs(self.X2_full[i]-self.x_sol_pos))
        self.x_index_neg = min(range(len(self.X2_full)), key = lambda i: abs(self.X2_full[i]-self.x_sol_neg))
        self.y_sol_pos = self.Y2_full[self.x_index_pos]
        self.y_sol_neg = self.Y2_full[self.x_index_neg] 
        if -self.parabola2_radius <= self.x_sol_pos <= self.parabola2_radius:
            self.x_sol = self.x_sol_pos
            self.y_sol = self.y_sol_pos
            self.index_sol = self.x_index_pos
        else:
            self.x_sol = self.x_sol_neg
            self.y_sol = self.y_sol_neg
            self.index_sol = self.x_index_neg
        self.collimation_x_index = self.index_sol
        self.collimation_XY = (self.x_sol, self.y_sol)
        self.collimation_gradient = abs(- 2 * (self.parabola2_depth / (self.parabola2_radius ** 2)) * self.collimation_XY[0]) #np.arctan(abs(self.Y2_full[self.collimation_x_index] \
                                                 #- self.Y2_full[self.collimation_x_index - 1 ]) / abs(self.X2_full[self.collimation_x_index] - self.X2_full[self.collimation_x_index - 1]) )
        self.collimation_angle = self.reflect_angle - 2 * self.collimation_gradient #angle to the right of the vertical axis
        #print(self.reflect_angle, self.collimation_gradient, self.collimation_angle)
        #self.collimation_angle_pos = -self.collimation_angle #make angle positive (mirror wrt to vertical axis)
        #print('reflect to coll', np.sqrt((self.reflect_xy[0] - self.collimation_XY[0])**2 + (self.reflect_xy[1] - self.collimation_XY[1])**2))
        #print('col ab',np.sqrt((self.reflect_xy[0] - self.collimation_XY[0])**2 + (self.reflect_xy[1] - self.collimation_XY[1])**2)*np.tan(7.337970883992557e-07) )
        return(self.collimation_XY, self.collimation_angle)

    def aperture(self, starting_location_x, collimation_XY, collimation_angle):
        self.collimation_XY = collimation_XY
        self.xy = collimation_XY
        self.angle = collimation_angle
        self.aperture_y = self.aperture_depth
        self.aperture_x = self.xy[0] + abs((self.aperture_y - self.xy[1])) * np.tan(self.angle)
        return(self.aperture_x,self.aperture_y)

    def trace_rays(self):
        self.path_list = []
        for j in self.ray_start():
            #print(j, datetime.datetime.now())
            self.j_index =  min(range(len(self.X1_full)), key = lambda i: abs(self.X1_full[i]-j))
            self.X1_slice = self.X1_full[int(self.j_index - (1 / self.step)): int(self.j_index + (1 / self.step))]
            self.Y1_slice = self.Y1_full[int(self.j_index - (1 / self.step)): int(self.j_index + (1 / self.step))]
            self.start = (j,0) #start point
            self.first = self.reflection(j, self.X1_slice, self.Y1_slice)#first reflection
            self.second = self.focal_abberation(j, self.first[0], self.first[1]) #through focal point
            self.third = self.collimation(j, self.first[0], self.first[1]) #second reflection
            self.fourth = self.aperture(j, self.third[0], self.third[1])#through aperture
            #print("time ended", datetime.datetime.now())
            self.path_list.append((self.start, self.first[0], self.second, self.third[0], self.fourth))
        return(self.path_list)
print('Start Time', datetime.datetime.now())


y = Parabolas(434,25,71.32,622.81, 0.01)
x1,y1 = y.parab_big()
x2,y2 = y.parab_small()
height = y.height_aperture()
print(height)
print("P", datetime.datetime.now())
rays =1
offset = (25/2) / ((434 / 2)/588.9295793606282) + 588.9295793606282

x = Raytrace(rays,434,25,71.32,offset,height[0],0.01,x1,y1,x2,y2)
fpcoord = x.FP_coord
coordinates = x.trace_rays()
print("End Time", datetime.datetime.now())
for i in np.arange(0,rays):
    coordinates_ray = coordinates[i]
    xlist = list(list(zip(*coordinates_ray))[0])
    #print(xlist[2])
    ylist = list(list(zip(*coordinates_ray))[1])
    plt.plot(xlist,ylist)
    #plt.annotate(("Ray",i),(0,0) ) 
aperturelist = (-39.71000000932372, 39.71000000932372)
apertureheightlist = (-70.40560018713194,-70.40560018713194)
plt.plot(x1,y1)
plt.plot(x2,y2)
plt.plot(aperturelist,apertureheightlist)
plt.annotate("Focal Point",fpcoord)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
