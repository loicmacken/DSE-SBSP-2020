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
#Quinten's Raptor engines
def cost_of_propulsions():
    total_cost = []
    #raptor engines
    raptor_engine = parts()        
    raptor_engine.set_value(1500000.,0)
    raptor_engine.transport_cost(1500, by_truck_high=50.11, by_ship_medium=14900) #bringing from US to Airbus
    raptor_engine.testing()
    total_cost.append(raptor_engine.cost_until_launch(40)*1.2)
    return sum(total_cost)
 
def cost_of_adcs():
    total_cost = []
    #sun sensor
    sun_sensor = parts()
    sun_sensor.set_value(12000.,1)
    sun_sensor.transport_cost(mass=4., by_truck_high=20.57)
    sun_sensor.testing()
    total_cost.append(sun_sensor.cost_until_launch(2)*1.2)
    #star tracker
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
    #thrusters
    thruster = parts()
    thruster.set_value(10000,1)
    thruster.transport_cost(4, by_truck_high=683.38)
    thruster.testing()
    total_cost.append(thruster.cost_until_launch(16)*1.2)
    return sum(total_cost)


def cost_of_cdh():
    pass

def cost_of_thermal():
     pass    
 
def cost_of_eps():
    pass

def cost_of_structures():
    pass

def cost_of_astro():
    pass

def cost_of_powerdl():
    pass

def cost_of_ass():
    pass
    
    
    

    
    
    
    
    
    
    
    
    