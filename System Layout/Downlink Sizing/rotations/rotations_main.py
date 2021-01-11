import numpy as np
from matrices import *


"""
x, y, z are the global axes, ie the body reference system, where x is the axis around which the system rolls,
y the axis around which it pitches and z is the axis around which it yaws

u, v, w related to body axis of the flat mirrors, u is perpendicular and facing outwards from the reflective surface,
v is parallel to the body y-axis (due to connection design) and w is perpendicular to both, facing in the 
positive direction of a rotation from u to v.

The x-z plane coincides with the ecliptic plane (ie the plane in which Earth orbits the sun),
the orbit of the system lies in the equatorial plane however, which is tilted by 23.44 degrees compared to the ecliptic plane.
The orientation of the equatorial plane wrt the system will propagate 360°/yr around the body y-axis, since the 
orientation of the equatorial wrt the ecliptic remains constant but the orientation aroud the body y-axis of the system
has to propagate 360° over the course of a year.
"""

ecl_plane = np.mat([[1,0,0],[0,0,1]]).T

ry = roty(-np.pi/2)
rz = rotz(-23.44 * np.pi/180)
equ_plane =  ry * rz * ecl_plane

sting = np.mat([[1,0,0],[0,1,0],[0,0,1]]).T # sting reference system
relay = roty(np.pi/2) * sting # relay reference system

#sting rotation test
yrot = roty(-np.pi/2)
wrot = rotz(-23.44 * np.pi/180)
vrot = roty(np.pi/6)

sti_ref= yrot * sting * wrot * vrot
# # test: first rotation about global y (pre), second rotation around local w (pre), these allign the fov with the equatorial plane,
# # third rotation around local v (post) to direct the beam to the desired location
# # alternatively you could do: yrot * zrot * vec * vrot, but this is impractical in reality due to the design of the three-axes rotational structure
#
# #a ray that started along the global x-axis now lies in the equatori plane: check if (a X b) . c = 0

check = check_in_plane(equ_plane, sti_ref[:,0])

print("Sting:\n",sti_ref, check)

# relay rotation test
yrot = roty(0*np.pi/4)
wrot = rotz(23.44 * np.pi/180)
vrot = roty(np.pi/3)

rel_ref = yrot * relay * wrot * vrot

check = check_in_plane(equ_plane, rel_ref[:,0])
print("Relay:\n", rel_ref, check)

#-----------------------------------------
# test with 2 rotations: 1 global, 1 local
#-----------------------------------------


