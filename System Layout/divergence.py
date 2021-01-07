import numpy as np
import time
import matplotlib.pyplot as plt
#planet
Mercury	= [57*10**9  ,9116.4]
Venus	= [108*10**9  ,	2611.0]
Earth	= [150*10**9  ,	1366.1]
Mars	= [227*10**9  ,	588.6]
Jupiter	= [778*10**9  ,	50.5]
Saturn	= [1426*10**9  , 15.04]
Uranus	= [2868*10**9  , 3.72]
Neptune	= [4497*10**9  , 1.51]
Pluto	= [5806*10**9  , 0.878]
planets = [Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto]
planet_names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto']
#divergence angle calculator for two planets

class Calculator:
    def __init__(self, planet1, planet2):
        #if(planet1[0]<planet2[0]):
        self.planet1 = planet1
        self.planet2 = planet2
        #else:
            #self.planet1 = planet2
            #self.planet2 = planet1
            
    def divergenceangle(self):
        dist = (self.planet2[0]-self.planet1[0])
        ratio = (self.planet1[1]/self.planet2[1]) #ratio in W/m2 between two planets
        radius_planet1 = np.sqrt(1/np.pi) #the radius of a circle with area 1 m2
        radius_planet2 = np.sqrt(ratio/np.pi) #the radius of a circle with area equal to the ratio to receive an equal amount of flux for 1m2 @planet 1
        height = radius_planet2-radius_planet1 #the difference in height between the two ratios
        angle = np.tan(height/dist)
        return(angle)
    
    def divergenceheight(self):
        dist = (self.planet2[0]-self.planet1[0])
        ratio = (self.planet1[1]/self.planet2[1]) #ratio in W/m2 between two planets
        radius_planet1 = np.sqrt(1/np.pi) #the radius of a circle with area 1 m2
        radius_planet2 = np.sqrt(ratio/np.pi) #the radius of a circle with area equal to the ratio to receive an equal amount of flux for 1m2 @planet 1
        return(radius_planet1, radius_planet2)



#combination of all planets

anglelist = []

for i in np.arange(0,len(planets)):
    planet1 = planets[i]
    for j in range(i+1,len(planets)):
        planet2 = planets[j]
        calc = Calculator(planet1,planet2)
        angle = calc.divergenceangle()
        anglelist.append([planet_names[i],planet_names[j],angle])
anglelist_just_angles = [x[2] for x in anglelist]
max_angle = max(anglelist_just_angles)
min_angle = min(anglelist_just_angles)
avg_angle = sum(anglelist_just_angles)/len(anglelist_just_angles)
#print(avg_angle*36*10**6)

#all possible combinations printed"""

"""
start_number = 0
it = 0
for i in range(0,len(planet_names)-1):
    print(planet_names[i])
    for j in range(start_number,(start_number+7-it+1)):
        print(anglelist[j])
        x = 1
    start_number = j+1
    it = it +1
    time.sleep(1)
"""

#ratio of solar flux wrt to distance

#Calculation of divergence angle through determining the trend line of the mercury angles

height_set = []
height_set.append(np.sqrt(1/np.pi))
for i in np.arange(1,len(planets)):
    planet1 = planets[0] 
    planet2 = planets[i] 
    calc = Calculator(planet1,planet2)
    planet1_height = calc.divergenceheight()[0]
    planet2_height = calc.divergenceheight()[1]
    height_set.append(planet2_height)
range_set = [x[0] for x in planets]

#plot these data as function of height
"""
plt.scatter( range_set, height_set)
plt.plot( range_set, height_set)
plt.show()
"""
#trendline
trend = np.polyfit( range_set, height_set,0)
#print(trend)

#angles
mercury_angles = anglelist_just_angles[0:8]
divergence_angle = np.average(mercury_angles)
