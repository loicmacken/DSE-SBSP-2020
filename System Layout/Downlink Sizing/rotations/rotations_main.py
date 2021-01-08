import numpy as np
from matrices import *


ecl_plane = np.array([[1,0,0],[0,0,1]]).T

rot23 = rotx(23.44)
equ_plane = rot23 * np.mat(ecl_plane)

print(equ_plane)

local = np.array([[1,0,0],[0,1,0],[0,0,1]]).T

yrot = roty(90)
wrot = rotz(23.44)
vrot = roty(10)

reflected = yrot * np.mat(local) * wrot * vrot
# test: first rotation about global z (pre), second rotation around global y (pre), these aling the fov with the equatorial plane,
# third rotaton around local v (post) to direct the beam to the desired location

#a ray that started along the global x-axis now lies in the equatorila plane: check if (a X b) . c = 0

cp = np.cross(equ_plane[:,0].T,equ_plane[:,1].T)
check = np.dot(cp,reflected[:,0])

print(check)