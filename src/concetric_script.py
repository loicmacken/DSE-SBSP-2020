import numpy as np
from math import *
from segment import Parabola
R = 434
n_segment = 1
r_segment = R/n_segment
rsegment = []
angle = []
def rsegment(n, segments):
    if n>=1:
        seg = segments[n] + rsegment((n-1), segments)
    else:
        return segments[n]
    return seg
def angle(n,angle1):
    if n >=1:
        angle_prev = angle((n-1),angle1)
    else:
        angle_prev = 0
    angle_now = angle1[n]
    return angle_prev + angle_now
def rcirclenproj(n, rsegmnent, alpha):
    if n>= 1:
        previous_r = rcirclenproj((n-1),segments[n-1], list_of_angles[n-1])
    else:
        previous_r = 49
    total = previous_r + segments[n]*cos(list_of_angles[n])
    return total
def lcirclen(n, rcirclen):
    return 2*pi*rcirclen

p = Parabola(434.0, 84.38, 25 + 14.71, 40)
pt = p.interpcurve()
p.calc_angles()
#print(p.angles)
list_of_angles = []
for n in range(39):
    list_of_angles.append(angle(n,p.angles))
#print(list_of_angles)


segments = p.get_segments()
segments = segments['lengths']

radius = []
for n in range(39):
    radius.append(rcirclenproj(n, segments, list_of_angles))
print(radius)