import numpy as np
"""
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
    print(i)"""
"""
x = [1,2,3]
y = 4
t = min(range(len(x)), key = lambda i: abs(x[i]-y))
print(t)"""

x = np.arange(0,10)
y = np.arange(0,10)
xy = []
for i in np.arange(len(x)):
    xy.append([x[i],y[i]])
print(xy)
#define parabola here, maybe import from parabola.py?

"""pv area = bus power
aperture lens
intake area
"""
print(np.gradient(x,y))