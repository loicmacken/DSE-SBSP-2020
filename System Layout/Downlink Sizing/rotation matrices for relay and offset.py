import numpy as np


ecl_plane = np.array([[1,0,0],[0,0,1]]).T

def rotx(degrees):
    cos = np.cos(np.radians(degrees))
    sin = np.sin(np.radians(degrees))
    Rx = np.mat([[1,0,0],[0,cos,-sin],[0,sin,cos]])
    return Rx

def roty(degrees):
    cos = np.cos(np.radians(degrees))
    sin = np.sin(np.radians(degrees))
    Ry = np.mat([[cos,0,sin],[0,1,0],[-sin,0,cos]])
    return Ry

def rotz(degrees):
    cos = np.cos(np.radians(degrees))
    sin = np.sin(np.radians(degrees))
    Rz = np.mat([[cos,-sin,0],[sin,cos,0],[0,0,1]])
    return Rz

def unit(vec):
    unitvec = np.array(vec)/np.sqrt(sum(np.array(vec)**2))
    return unitvec

rot23 = rotx(23.44)
equ_plane = rot23 * np.mat(ecl_plane)

print(equ_plane)

local = np.array([[1,0,0],[0,1,0],[0,0,1]]).T

yrot = roty(90)
zrot = rotz(23.44)
vrot = roty(10)

reflected = yrot * zrot * np.mat(local) * vrot
# test: first rotation about global z (pre), second rotation around global y (pre), these aling the fov with the equatorial plane,
# third rotaton around local v to direct the beam to the desired location

#a ray along the global x-axis now lies in the equatorila plane: check if (a X b) . c = 0

cp = np.cross(equ_plane[:,0].T,equ_plane[:,1].T)
check = np.dot(cp,reflected[:,0])

print(check)