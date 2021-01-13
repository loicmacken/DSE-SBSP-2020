# new model to estimate the cost for a part

import numpy as np

class parts:
    
    def __init__(self):
        self.transport = []
        self.development = []
        
    def set_value(self, value, EU):
        self.value = value
        self.EU = EU
        if EU == 0:
            self.value = value #add import taxes
            self.import_taxes = value*0.064
            self.transport.append(self.import_taxes)
            
    def transport_cost(self, mass, by_air=0, by_truck_low=0, by_truck_high=0, by_train=0, by_ship_small=0, by_ship_medium=0, by_ship_large=0):     
        self.mass = mass
        air = 0.18/1000
        truck_low = 0.092 * 1.338/1000
        truck_high = 0.189 * 1.338/1000
        train = 0.017/1000
        ship_small = 0.013/1000
        ship_medium = 0.033/1000
        ship_large = 0.023/1000
        cost_breakdown = []
        mass = mass*1.25
        cost_breakdown.append(mass*air*by_air)
        cost_breakdown.append(mass*truck_low*by_truck_low)
        cost_breakdown.append(mass*truck_high*by_truck_high)
        cost_breakdown.append(mass*train*by_train)
        cost_breakdown.append(mass*ship_small*by_ship_small)
        cost_breakdown.append(mass*ship_medium*by_ship_medium)
        cost_breakdown.append(mass*ship_large*by_ship_large)
        self.transport.append(sum(cost_breakdown))
    
    def part_cost(self, material_cost, machining_cost, development_cost, labour):
        self.development.append(development_cost) 
        self.value = material_cost + machining_cost + labour
        self.EU = 0
        
    def testing(self):
        self.testing_cost = self.value*0.03 + sum(self.development)*0.3
        self.development.append(self.testing_cost)

    def cost_until_launch(self, units):
        trans = sum(self.transport)
        dev = sum(self.development)
        self.total  = trans + dev + self.value
        return self.total*units
        
    
    


    
#Propulsionssss


total_cost = []
#raptor engines SpaceX Hawthorne US of A
raptor_engine = parts()        
raptor_engine.set_value(1500000.,0)
raptor_engine.transport_cost(1500, by_truck_high=50.11, by_ship_medium=14900) #bringing from US to Airbus
raptor_engine.testing()
total_cost.append(raptor_engine.cost_until_launch(40)*1.2)
#propellant tank made in house
prop_tank = parts()
prop_tank.part_cost(80000, 100000, 2000000/40, 260000)
prop_tank.testing()
total_cost.append(prop_tank.cost_until_launch(40)*1.2)
 
#ADCS parts
#sun sensor Netherlands
sun_sensor = parts()
sun_sensor.set_value(12000.,1)
sun_sensor.transport_cost(mass=4., by_truck_high=20.57)
sun_sensor.testing()
total_cost.append(sun_sensor.cost_until_launch(2)*1.2)
#star tracker Sagitta Belgium Antwerp Area
star_tracker = parts()
star_tracker.set_value(45000,1)
star_tracker.transport_cost(4, by_truck_high=106.2)
star_tracker.testing()
total_cost.append(star_tracker.cost_until_launch(2)*1.2)
#imu 
imu = parts()
imu.set_value(1700, 1)
imu.transport_cost(4, by_truck_high=1214)
imu.testing()
total_cost.append(imu.cost_until_launch(2)*1.2)
#thrusters Ariane Group GmbH Germany
thruster = parts()
thruster.set_value(10000,1)
thruster.transport_cost(4, by_truck_high=683.38)
thruster.testing()
total_cost.append(thruster.cost_until_launch(16)*1.2)
#propellant (made in house)
adcs_tank = parts()
adcs_tank.part_cost(8000, 10000, 50000/16, 26000)
adcs_tank.testing()
total_cost.append(adcs_tank.cost_until_launch(16)*1.2)



#C&DH
#computer 
computer = parts()
computer.set_value(7500, 1)
computer.transport_cost(20, by_truck_high=1214)
computer.testing()
total_cost.append(computer.cost_until_launch(1)*1.2)
    
    

#Thermal parts
#LDR
thermal_LDR = parts()
thermal_LDR.part_cost(25*2200+5*13000, 1000000, 10000000/2, 15*1*75000)
thermal_LDR.testing()
total_cost.append(thermal_LDR.cost_until_launch(2)*1.2)
#Thermal straps azimut space berlin
thermal_straps = parts()
thermal_straps.set_value(2000, 1)
thermal_straps.transport_cost(50, by_truck_high=615.52)
thermal_straps.testing()
total_cost.append(thermal_straps.cost_until_launch(25)*1.2)
#coating ATG Europe BV noordwijk
thermal_coat = parts()
thermal_coat.set_value(8000, 1)
thermal_coat.transport_cost(4, by_truck_high=50.21)
thermal_coat.testing()
total_cost.append(thermal_coat.cost_until_launch(2)*1.2)
#MLI in house 
thermal_MLI = parts()
thermal_MLI.part_cost(500*2200, 500*22, 100000, 15*3*52000)
thermal_MLI.testing()
total_cost.append(thermal_MLI.cost_until_launch(1)*1.2)
#Louvres inhouse
thermal_louvres = parts()
thermal_louvres.part_cost(4*2000, 4*2000, 10000, 3*3*52000)
thermal_louvres.testing()
total_cost.append(thermal_louvres.cost_until_launch(1)*1.2)
#heating element by RS components Netherlands
thermal_heating = parts()
thermal_heating.set_value(1000, 1)
thermal_heating.transport_cost(mass=10, by_truck_high=30.06)
thermal_heating.testing()
total_cost.append(thermal_heating.cost_until_launch(40+16)*1.2)

 
#EPS
#batteries
#wiring
#solar panels

def cost_of_structures():
    pass

def cost_of_astro():
    pass

def cost_of_powerdl():
    pass

def cost_of_ass():
    pass
    
    
    
def final_pre_launch():
    pass

    
    
print(sum(total_cost)/100000)  
    
    
    
    
    
    