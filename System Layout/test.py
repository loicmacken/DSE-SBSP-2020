import numpy as np

radius = 10
rays = 8
steps = radius/(rays+1)
liststep = []
print(steps)
for i in np.arange(0+steps,radius,steps):

    liststep.append(i)
print(liststep)
print(len(liststep))
""""
for i in np.arange(0,10):
    print(i)"""