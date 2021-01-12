import numpy as np
import matplotlib.pyplot as plt
"""
#self.intersect_angled_gradient = (np.arctan( (self.Y1[self.intersect_angled_index] \
                                                    - self.Y1[self.intersect_angled_index - 1]) / (self.X1[self.intersect_angled_index] - self.X1[self.intersect_angled_index - 1]) )) #same thing as unangled gradient
        self.intersect_angled_angle = self.divergence + 2 * (self.intersect_angled_gradient - (self.divergence))
if (self.ray_amount % 2) == 0:
            self.rays_left = self.ray_amount / 2
            self.rays_right = self.ray_amount / 2
        else:
            self.rays_left = (self.ray_amount + 1) / 2
            self.rays_right = (self.ray_amount - 1) / 2
        
for self.j in np.arange(self.parabola1_radius - self.step_size_right, self.parabola1_radius-self.intake_right, -self.step_size_right):
            self.starting_locations_right.append(self.j)
                    #self.step_size_right = self.intake_right / (self.rays_right + 1)
radius = 10
rays = 8
steps = radius/(rays+1)
liststep = []
print(steps)
for i in np.arange(0+steps,radius,steps):

    liststep.append(i)
print(liststep)
print(len(liststep))

for i in np.arange(0,10):
    print(i)

x = [1,2,3]
y = 4
t = min(range(len(x)), key = lambda i: abs(x[i]-y))
print(t)

x = np.arange(0,10)
y = np.arange(0,10)
xy = []
for i in np.arange(len(x)):
    xy.append([x[i],y[i]])
print(xy)
#define parabola here, maybe import from parabola.py?

pv area = bus power
aperture lens
intake area

print(np.gradient(x,y))"""
"""
class test:
    def __init__(self, val):
        self.val = val
    def testfunc(self):
        x =1
    def testfunc2(self):
        return(x)
a = test(1)
print(a.testfunc2())


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
        self.FP_coord = [self.parabola1_radius / 2, self.FP - self.parabola1_depth]
        self.divergence = divergence_angle
        self.parabola2_depth = self.parabola2_radius * (self.parabola1_depth / self.parabola1_radius)

    def parab_big(self):
        self.X = np.arange(-self.parabola1_radius, self.parabola1_radius + self.step, self.step)
        self.Y = (-1) ** self.up * self.parabola1_depth * (1 - self.X ** 2 / self.parabola1_radius ** 2)
        self.length = np.sqrt(self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) \
                      + self.parabola1_radius ** 2 * np.arcsinh(2 * self.parabola1_depth / self.parabola1_radius) / (2 * self.parabola1_depth)
        self.A = np.pi * self.parabola1_radius / (6 * self.parabola1_depth ** 2) * ((self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) ** (3 / 2) - self.parabola1_radius ** 3)
        return(self.X, self.Y)
    
    def parab_small(self):
        self.X2 = np.arange(-self.parabola2_radius, self.parabola2_radius + self.step, self.step)
        self.Y2 = self.parabola2_depth * (1 - self.X2 ** 2 / self.parabola2_radius ** 2)
        return(self.X2,self.Y2)
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
   
    def reflection(self, starting_location_x): #first calculate the intersect point if beam straight, then calculate intersect point if divergent, then calculate angle towards focal point
        self.starting_location_x = starting_location_x #the x location where the ray begins
        self.intersect_unangled_index = min(range(len(self.parab_big()[0])), key = lambda i: abs(self.parab_big()[0][i]-self.starting_location_x)) #the index of the number in the list of parabola x values that is closest to the start x location of the ray
        self.intersect_unangled_local_gradient =  (np.pi / 4) - (np.arctan( (self.parab_big()[1][self.intersect_unangled_index] \
                                                    - self.parab_big()[1][self.intersect_unangled_index - 1]) / (self.parab_big()[0][self.intersect_unangled_index] - self.parab_big()[0][self.intersect_unangled_index - 1]) )) #the local gradient of the parabola between the closest x and the previous x value
        self.intersect_unangled_distance_y = abs(self.parab_big()[1][self.intersect_unangled_index])
        self.intersect_angled_x = self.starting_location_x - np.tan(self.divergence + self.parabola1_angle) * self.intersect_unangled_distance_y \
                                     * (np.tan(self.intersect_unangled_local_gradient) / (np.tan(self.intersect_unangled_local_gradient) + np.tan(self.divergence + self.parabola1_angle)) ) #calculate the x coordinate of the angled beam through some geometry calculations-> dist=wtana/(tana+tanb)
        self.intersect_angled_index = min(range(len(self.parab_big()[0])), key = lambda i: abs(self.parab_big()[0][i]-self.intersect_angled_x))  #same story as other index but now for angled intersect
        self.intersect_angled_XY =  [self.parab_big()[0][self.intersect_angled_index], self.parab_big()[1][self.intersect_angled_index] ] #xy coordinate of the intersect point if the beam is divergent
        self.intersect_angled_gradient = (np.arctan( (self.parab_big()[1][self.intersect_angled_index] \
                                                    - self.parab_big()[1][self.intersect_angled_index - 1]) / (self.parab_big()[0][self.intersect_angled_index] - self.parab_big()[0][self.intersect_angled_index - 1]) )) #same thing as unangled gradient
        self.intersect_angled_angle = self.divergence + self.parabola1_angle + 2 * (self.intersect_angled_gradient - (self.divergence + self.parabola1_angle)) #angle of beam wrt vertical I/
        return(self.intersect_angled_XY,self.intersect_angled_angle)

    def focal_abberation(self, starting_location_x):
        self.starting_location_x = starting_location_x
        print("commence")
        self.reflect_xy = self.reflection(self.starting_location_x)[0]
        print("halfway")
        self.reflect_angle = self.reflection(self.starting_location_x)[1]
        self.focal_y = self.FP_coord[1]
        
        self.abberation_y = self.focal_y #- self.reflect_xy[1]
        self.abberation_x = self.reflect_xy[0] + (self.abberation_y -self.reflect_xy[1]) * np.tan(self.reflect_angle)
        return(self.abberation_x, self.abberation_y)





x = Raytrace(20,400,10,71,0.01)
print(x.focal_abberation(100))


"""
"""
xval = x.parabola()[0]
yval = x.parabola()[1]
plt.plot(xval,yval)
plt.axis('equal')
plt.show()"""
"""self.intersect_angled_x = self.starting_location_x - np.tan(self.divergence + self.parabola1_angle) * self.intersect_unangled_distance_y \
                                     * (np.tan(self.intersect_unangled_local_gradient) / (np.tan(self.intersect_unangled_local_gradient) + np.tan(self.divergence + self.parabola1_angle)) )#calculate the x coordinate of the angled beam through some geometry calculations-> dist=wtana/(tana+tanb)
        self.intersect_angled_index = min(range(len(self.X1)), key = lambda i: abs(self.X1[i]-self.intersect_angled_x)) #same story as other index but now for angled intersect"""
"""
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
        print("done with parabola")
        self.length = np.sqrt(self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) \
                      + self.parabola1_radius ** 2 * np.arcsinh(2 * self.parabola1_depth / self.parabola1_radius) / (2 * self.parabola1_depth)
        self.A = np.pi * self.parabola1_radius / (6 * self.parabola1_depth ** 2) * ((self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) ** (3 / 2) - self.parabola1_radius ** 3)
        return(self.X1, self.Y1)

    def parab_small(self):
        self.X2 = np.arange(-self.parabola2_radius, self.parabola2_radius + self.step, self.step)
        self.Y2 = self.parabola2_depth * (1 - self.X2 ** 2 / self.parabola2_radius ** 2)
        self.Y2 = self.Y2 + self.parabola2_offset
        return(self.X2,self.Y2)

y = Parabolas(434,25,71,622.81, 0.001)
x1,y1 = y.parab_big()
x2,y2 = y.parab_small()
blocked = 25+14.71
h = min(range(len(x1)), key = lambda i: abs(x1[i]-blocked))
print(x1[h],y1[h])"""

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
        self.length = np.sqrt(self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) \
                      + self.parabola1_radius ** 2 * np.arcsinh(2 * self.parabola1_depth / self.parabola1_radius) / (2 * self.parabola1_depth)
        self.A = np.pi * self.parabola1_radius / (6 * self.parabola1_depth ** 2) * ((self.parabola1_radius ** 2 + 4 * self.parabola1_depth ** 2) ** (3 / 2) - self.parabola1_radius ** 3)
        return(self.X1, self.Y1)

    def parab_small(self):
        self.X2 = np.arange(-self.parabola2_radius, self.parabola2_radius + self.step, self.step)
        self.Y2 = self.parabola2_depth * (1 - self.X2 ** 2 / self.parabola2_radius ** 2)
        self.Y2 = self.Y2 + self.parabola2_offset
        return(self.X2,self.Y2)
    
    def height_aperture(self):
        self.radius_flat = self.parabola2_radius + self.pv_width
        self.x = self.parab_big()[0]
        #print(len(self.x))
        self.height_index = min(range(len(self.x)), key = lambda i: abs(self.x[i]-self.radius_flat))
        self.height = self.parab_big()[1][self.height_index]
        return(self.height)
y = Parabolas(434,25,71.32,622.81, 0.001)
x1,y1 = y.parab_big()
x2,y2 = y.parab_small()
height = y.height_aperture()
x = [(-236.85500000466186, -49.85323184325536), (-1.215240385177026, 588.9295793606282), (10.992297920908012, 626.1092158023899), (58.18951410045132, -70.40560018774643)]
xlist = list(list(zip(*x))[0])
ylist = list(list(zip(*x))[1])
#print(len(x2))
startx = -100
refx =  min(range(len(x1)), key = lambda i: abs(x1[i]-startx))
vector1 = np.array([0,y1[refx]])
grad = -(-2 * 71.32/(434**2)) * -100
vector2 = np.array([1,grad])
vector2n = np.array([-grad, 1])
vector3 = vector1 - 2*vector2n*np.dot(vector1,vector2n)
V = np.array([vector1, vector2, vector3])
origin = (x1[refx],y1[refx]) # origin point
plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
plt.show()
"""
plt.plot(xlist,ylist)
plt.plot(x1,y1)
plt.annotate(("Ray","hello"),(0,0) )
plt.plot(x2,y2)
plt.show()"""
"""
x = np.array([1,2])
y = np.array([-2,1])
print(np.dot(x,y))"""
