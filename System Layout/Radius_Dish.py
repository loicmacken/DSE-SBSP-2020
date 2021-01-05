import numpy as np

# INPUTS
A_pv = 5*10**6 / 1362 # 5MW for bus power from pv cells
r_beam = 10
A_dish = 863*10**6 / 1362 # 800 MW for payload power by parabolic dish


"""apperture will be a disk-shaped hole in the center of the main dish with radius r_beam. 
pv ring is like a flat Saturn disk around the apperture.
we need to substract the radii from the main dish radius

we have: A_pv = pi*(r_pv + r_beam)² - pi*r_beam²      
--> rework to find r_pv and we get:     pi*r_pv² + 2pi*r_beam*r_pv - A_pv = 0
which is a simple quadratic function"""

root_D = np.sqrt(4 * r_beam**2 * np.pi**2 + 4 * np.pi * A_pv) # quadratic : a = pi, b= 2pi*r_beam, c = -A_pv
r1 = (-2*np.pi*r_beam + root_D)/(2*np.pi)
r2 = (-2*np.pi*r_beam - root_D)/(2*np.pi)

r_pv = max(r1,r2)

"""Same procedure for the effective intake radius of the main dish
A_dish = pi*(r_dish + r_pv + r_beam)² - pi*(r_pv + r_beam)² 
so:     pi*r_dish**2 + 2pi*r_dish*(r_pv + r_beam) - A_dish = 0"""

a = np.pi
b = 2*np.pi*(r_pv + r_beam)
c = -A_dish
root_D = np.sqrt(b**2 - 4*a*c)

r1 = (-b + root_D)/(2*a)
r2 = (-b - root_D)/(2*a)

r_dish = max(r1,r2)

r_total = r_beam + r_pv + r_dish

print(r_beam, r_pv, r_dish, r_total)