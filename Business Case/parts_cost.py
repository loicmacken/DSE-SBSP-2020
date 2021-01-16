# new model to estimate the cost for a part
import matplotlib.pyplot as plt
import numpy as np


def chartify(num, tot):
    angle = num/tot * 360
    return angle



class parts:
    
    def __init__(self, name):
        self.transport = []
        self.development = []
        self.name = name
        self.status = 'In-House'
        self.import_tax = 'None'
        
    def set_value(self, value, EU):
        self.value = value
        self.EU = EU
        self.status = 'Off-The-Shelve'
        if EU == 0:
            self.value = value #add import taxes
            self.import_taxes = value*0.064
            self.import_tax = '6.4\%'
            self.transport.append(self.import_taxes)
        return self.value
    
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
        return sum(cost_breakdown)
    
    def part_cost(self, material_cost, machining_cost, development_cost, labour):
        self.part_costs = material_cost + machining_cost
        self.development.append(development_cost)
        self.development_costs = development_cost
        self.value = material_cost + machining_cost + labour
        self.EU = 0
        
    def testing(self, factor=1):
        self.testing_cost = (self.value*0.03 + sum(self.development)*0.3)*factor
        self.development.append(self.testing_cost)
        return self.testing_cost

    def cost_until_launch(self, units):
        self.units = units
        trans = sum(self.transport)
        dev = sum(self.development)
        self.total  = np.round((trans + dev + self.value), 2)
        return self.total*units
        
    def __str__(self):
        return print(f'{self.name} & {self.status} & {self.import_tax} & {self.total} & {self.units} \\\ ' )
    
####global_variables#####
launches = 201
years = 25

    
#Propulsionssss


total_cost = []
total_mass = []

rd_cost = [] #done
cost_of_parts = [] #done
cost_of_transport = [] #done
cost_of_manufacturing = [] #done
cost_of_launches = [] #done

#raptor engines SpaceX Hawthorne US of A
raptor_engine = parts('raptor engine')        
raptor_engine.set_value(1500000.,0)
raptor_engine.transport_cost(1500, by_truck_high=50.11, by_ship_medium=14900) #bringing from US to Airbus
cost_of_transport.append(40*raptor_engine.transport_cost(1500, by_truck_high=50.11, by_ship_medium=14900))
raptor_engine.testing()
rd_cost.append(raptor_engine.testing())
cost_of_parts.append(raptor_engine.set_value(1500000.,0)*40)
total_cost.append(raptor_engine.cost_until_launch(40)*1.2)
raptor_engine.__str__()
#propellant tank made in house
prop_tank = parts('propellant tank')
prop_tank.part_cost(80000, 100000, 2000000/40, 260000)
prop_tank.testing()
rd_cost.append(prop_tank.testing())
rd_cost.append(prop_tank.development_costs)
cost_of_manufacturing.append(prop_tank.part_costs*40)
total_cost.append(prop_tank.cost_until_launch(40)*1.2)
prop_tank.__str__()
 
#ADCS parts
#sun sensor Netherlands
sun_sensor = parts('sun sensor')
sun_sensor.set_value(12000.,1)
sun_sensor.transport_cost(mass=4., by_truck_high=20.57)
cost_of_transport.append(2*sun_sensor.transport_cost(mass=4., by_truck_high=20.57))
sun_sensor.testing()
rd_cost.append(sun_sensor.testing())
cost_of_parts.append(sun_sensor.set_value(12000.,1)*2)
total_cost.append(sun_sensor.cost_until_launch(2)*1.2)
sun_sensor.__str__()
#star tracker Sagitta Belgium Antwerp Area
star_tracker = parts('star tracker')
star_tracker.set_value(45000,1)
star_tracker.transport_cost(4, by_truck_high=106.2)
cost_of_transport.append(2*star_tracker.transport_cost(4, by_truck_high=106.2))
star_tracker.testing()
rd_cost.append(star_tracker.testing())
cost_of_parts.append(star_tracker.set_value(45000,1)*2)
total_cost.append(star_tracker.cost_until_launch(2)*1.2)
star_tracker.__str__()
#imu 
imu = parts('imu')
imu.set_value(1700, 1)
imu.transport_cost(4, by_truck_high=1214)
cost_of_transport.append(2*imu.transport_cost(4, by_truck_high=1214))
imu.testing()
rd_cost.append(imu.testing())
cost_of_parts.append(imu.set_value(1700, 1)*2)
total_cost.append(imu.cost_until_launch(2)*1.2)
imu.__str__()
#thrusters Ariane Group GmbH Germany
thruster = parts('ADCS thruster')
thruster.set_value(10000,1)
thruster.transport_cost(4, by_truck_high=683.38)
cost_of_transport.append(16*thruster.transport_cost(4, by_truck_high=683.38))
thruster.testing()
rd_cost.append(thruster.testing())
cost_of_parts.append(thruster.set_value(10000,1)*16)
total_cost.append(thruster.cost_until_launch(16)*1.2)
thruster.__str__()
#propellant (made in house)
adcs_tank = parts('ADCS tank')
adcs_tank.part_cost(8000, 10000, 50000/16, 26000)
adcs_tank.testing()
rd_cost.append(adcs_tank.testing())
rd_cost.append(adcs_tank.development_costs*16)
cost_of_manufacturing.append(adcs_tank.part_costs*16)
total_cost.append(adcs_tank.cost_until_launch(16)*1.2)
adcs_tank.__str__()


#C&DH
#computer 
computer = parts('On-Board computer')
computer.set_value(7500, 1)
computer.transport_cost(20, by_truck_high=1214)
cost_of_transport.append(computer.transport_cost(20, by_truck_high=1214))
computer.testing()
rd_cost.append(computer.testing())
cost_of_parts.append(computer.set_value(7500, 1))
total_cost.append(computer.cost_until_launch(1)*1.2)
computer.__str__()
#antennae
antenna = parts('Antenna')
antenna.set_value(2000, 1)
antenna.transport_cost(50, by_truck_high=615.52)
cost_of_transport.append(antenna.transport_cost(20, by_truck_high=1214))
antenna.testing()
rd_cost.append(antenna.testing())
cost_of_parts.append(antenna.set_value(2000, 1)*3)
total_cost.append(antenna.cost_until_launch(3)*1.2)
antenna.__str__()
    

#Thermal parts
#LDR
thermal_LDR = parts('Liquid Droplet Radiator')
thermal_LDR.part_cost(25*2200+5*13000, 1000000, 10000000/2, 15*1*75000)
thermal_LDR.testing()
rd_cost.append(thermal_LDR.testing())
rd_cost.append(thermal_LDR.development_costs*2)
cost_of_manufacturing.append(thermal_LDR.part_costs*2)
total_cost.append(thermal_LDR.cost_until_launch(2)*1.2)
thermal_LDR.__str__()
#Thermal straps azimut space berlin
thermal_straps = parts('Thermal straps')
thermal_straps.set_value(2000, 1)
thermal_straps.transport_cost(50, by_truck_high=615.52)
cost_of_transport.append(50*thermal_straps.transport_cost(50, by_truck_high=615.52))
thermal_straps.testing()
rd_cost.append(thermal_straps.testing())
cost_of_parts.append(thermal_straps.set_value(2000, 1)*25)
total_cost.append(thermal_straps.cost_until_launch(25)*1.2)
thermal_straps.__str__()
#coating ATG Europe BV noordwijk
thermal_coat = parts('Thermal coatings')
thermal_coat.set_value(8000, 1)
thermal_coat.transport_cost(4, by_truck_high=50.21)
cost_of_transport.append(2*thermal_coat.transport_cost(4, by_truck_high=50.21))
thermal_coat.testing()
rd_cost.append(thermal_coat.testing())
cost_of_parts.append(thermal_coat.set_value(8000, 1)*2)
total_cost.append(thermal_coat.cost_until_launch(2)*1.2)
thermal_coat.__str__()
#MLI in house 
thermal_MLI = parts('Multi-layer Insulation')
thermal_MLI.part_cost(500*2200, 500*22, 100000, 15*3*52000)
thermal_MLI.testing()
rd_cost.append(thermal_MLI.testing())
rd_cost.append(thermal_MLI.development_costs)
cost_of_manufacturing.append(thermal_MLI.part_costs)
total_cost.append(thermal_MLI.cost_until_launch(1)*1.2)
thermal_MLI.__str__()
#Louvres inhouse
thermal_louvres = parts('Louvres')
thermal_louvres.part_cost(4*2000, 4*2000, 10000, 3*3*52000)
thermal_louvres.testing()
rd_cost.append(thermal_louvres.testing())
rd_cost.append(thermal_louvres.development_costs)
cost_of_manufacturing.append(thermal_louvres.part_costs)
total_cost.append(thermal_louvres.cost_until_launch(1)*1.2)
thermal_louvres.__str__()
#heating element by RS components Netherlands
thermal_heating = parts('Heating elements')
thermal_heating.set_value(1000, 1)
thermal_heating.transport_cost(mass=10, by_truck_high=30.06)
cost_of_transport.append(56*thermal_heating.transport_cost(mass=10, by_truck_high=30.06))
thermal_heating.testing()
rd_cost.append(thermal_heating.testing())
cost_of_parts.append(thermal_heating.set_value(1000, 1)*56)
total_cost.append(thermal_heating.cost_until_launch(40+16)*1.2)
thermal_heating.__str__()

 
#EPS
#batteries
solar_panels = parts('Solar Panels')
solar_panels.set_value(400000, 1)
solar_panels.transport_cost(50, by_truck_high=615.52)
cost_of_transport.append(50*solar_panels.transport_cost(50, by_truck_high=615.52))
solar_panels.testing()
rd_cost.append(solar_panels.testing())
cost_of_parts.append(solar_panels.set_value(40000, 1)*25)
total_cost.append(solar_panels.cost_until_launch(2)*1.2)
solar_panels.__str__()
#solar panels
Batteries = parts('Battery packs')
Batteries.set_value(35000, 0)
Batteries.transport_cost(50, by_truck_high=615.52)
cost_of_transport.append(50*Batteries.transport_cost(50, by_truck_high=615.52))
Batteries.testing()
rd_cost.append(Batteries.testing())
cost_of_parts.append(Batteries.set_value(35000, 1)*25)
total_cost.append(Batteries.cost_until_launch(2)*1.2)
Batteries.__str__()
#Power management system
PDS = parts('Power Distribution System')
PDS.set_value(10000, 0)
PDS.transport_cost(50, by_truck_high=615.52)
cost_of_transport.append(50*PDS.transport_cost(50, by_truck_high=615.52))
PDS.testing()
rd_cost.append(PDS.testing())
cost_of_parts.append(PDS.set_value(10000, 1)*25)
total_cost.append(PDS.cost_until_launch(2)*1.2)
PDS.__str__()

#Astrophysics section
#fuel for transfers
transfer_fuel = parts('Transfer fuel')
transfer_fuel.part_cost(10000000, 5000, 0, 52000)
transfer_fuel.testing()
rd_cost.append(transfer_fuel.testing())
rd_cost.append(transfer_fuel.development_costs)
cost_of_manufacturing.append(transfer_fuel.part_costs)
total_cost.append(transfer_fuel.cost_until_launch(1)*1.2)
transfer_fuel.__str__()

#structures & downlink costs
#Trusses
truss = parts('Truss-structures')
truss.part_cost(3.2*2820643, 2.5*2820643, 0.5*2820643, 90*52000 )
truss.testing()
rd_cost.append(truss.testing())
rd_cost.append(truss.development_costs)
cost_of_manufacturing.append(truss.part_costs)
total_cost.append(truss.cost_until_launch(1)*1.2)
truss.__str__()
#Queen - needs to be made twice in total
queen = parts("Queen's 'Mirrors")
queen.part_cost(3.2*8555862.789, 2.5*8555862.789, 0.5*8555862.789, 20*52000 )
queen.testing()
rd_cost.append(queen.testing())
rd_cost.append(queen.development_costs)
cost_of_manufacturing.append(queen.part_costs)
total_cost.append(queen.cost_until_launch(1)*1.2)
queen.__str__()
#worker - needs to be made twice in total
worker = parts("Worker's 'Mirrors")
worker.part_cost(3.2*30509.1, 6.5*30509.1, 30509.1, 5*52000 )
worker.testing()
rd_cost.append(worker.testing())
rd_cost.append(worker.development_costs)
cost_of_manufacturing.append(worker.part_costs)
total_cost.append(worker.cost_until_launch(1)*1.2)
worker.__str__()
#stinger - needs to be made twice in total
stinger = parts("Stinger's 'Mirrors")
stinger.part_cost(3.2*45614.4, 6.5*45614.4, 45614.4, 5*52000 )
stinger.testing()
rd_cost.append(stinger.testing())
rd_cost.append(stinger.development_costs)
cost_of_manufacturing.append(stinger.part_costs)
total_cost.append(stinger.cost_until_launch(1)*1.2)
stinger.__str__()
#relay - needs to be made twice in total
relay = parts("Relay's 'Mirrors")
relay.part_cost(1.6*272956.95, 3.3*272956.95, 0.25*272956.95, 5*52000 )
relay.testing()
rd_cost.append(relay.testing())
rd_cost.append(relay.development_costs)
cost_of_manufacturing.append(relay.part_costs)
total_cost.append(relay.cost_until_launch(1)*1.2)
relay.__str__()

    


#cost of assembly 
#robots - Darpa Phoenix
ass_robot = parts('Assembly robot')
ass_robot.set_value(64200000, 0)
ass_robot.transport_cost(1500, by_truck_high=50.11, by_ship_medium=14900) #bringing from US to Airbus
cost_of_transport.append(3*ass_robot.transport_cost(1500, by_truck_high=50.11, by_ship_medium=14900))
ass_robot.testing(10)
rd_cost.append(ass_robot.testing(10))
cost_of_parts.append(ass_robot.set_value(64200000, 0)*3)
total_cost.append(ass_robot.cost_until_launch(3)*1.2)
ass_robot.__str__()

#mechanisms and connections
mechs = parts('Mechanisms')
mechs.part_cost(50000,5000 , 1000000/300, 52000)
mechs.testing()
rd_cost.append(mechs.testing())
rd_cost.append(mechs.development_costs*300)
cost_of_manufacturing.append(mechs.part_costs*300)
total_cost.append(mechs.cost_until_launch(300)*1.2)
mechs.__str__()


#moving everything cost
moving = parts('Module transport')
moving.set_value(0, 1)
moving.transport_cost(20200000, by_truck_high=150.11, by_ship_large=7500)
total_cost.append(moving.cost_until_launch(1)*1.2)
cost_of_transport.append(moving.transport_cost(40000000, by_truck_high=150.11, by_ship_large=7500))
moving.__str__()
#maintenance cost:
#maintain parts missions
timespan = 25*12
mirr = 0.999
sol = 0.9985
mirrors_1 = [100, 100, 100, 100]
sols_1 = [100]
def copy(lst1):
    lst = []
    for i in lst1:
        lst.append(i)
    return lst

mainmissionmirrs = 0
mainmissionsols = 0
mirrors = copy(mirrors_1)
sols = copy(sols_1)
for month in range(0,timespan):
    for i in range(0, len(mirrors)):
        mirrors[i] = mirrors[i]*mirr
        
    for x in range(0, len(sols)):
        sols[x] = sols[x]*sol
        
    outputmirrs = np.average(mirrors)
    outputsols = np.average(sols)
    if outputmirrs < 94:
        for i in range(len(mirrors)):
            if outputmirrs < 94:
                a = min(mirrors)
                b = mirrors.index(a)
                mirrors[b] = mirrors_1[b]
                mainmissionmirrs += 1
                outputmirrs = np.average(mirrors)
            else:
                continue
    
    if outputsols < 80:
        for i in range(len(sols)):
            if outputsols < 80:
                a = min(sols)
                b = sols.index(a)
                sols[b] = sols_1[b]
                mainmissionsols += 1
                outputsols = np.average(sols)
            else:
                
                continue
    print(month, mainmissionmirrs, mainmissionsols)    
   

#operating costs:
#salaries
workers = 350
average_energy_salary = 52000 #including bonusses ofc
cost_salaries = workers*average_energy_salary*years
total_cost.append(cost_salaries)
#facility
ground_station = parts('Ground station')
ground_station.part_cost(500000000, 0, 0, 0)
ground_station.testing(0)
total_cost.append(ground_station.cost_until_launch(1)*1.2)
ground_station.__str__()

#The EOL cost is integrated into operating and fuel costs
#launch costs:
cost_per_lv = 30000000
launches += mainmissionsols + mainmissionmirrs
launch_cost = launches * cost_per_lv    
total_cost.append(launch_cost)
cost_of_launches.append(launch_cost)    
    



total_cost = sum(rd_cost+cost_of_launches+cost_of_transport+cost_of_parts+[cost_salaries])
    
print('=====================================')
print('cost of development:            ' + str(sum(rd_cost)/1000000))
print('cost of launches:               ' + str(sum(cost_of_launches)/1000000))
print('cost of transport:              ' + str(sum(cost_of_transport)/1000000))   
print('cost of purchases:              ' + str(sum(cost_of_parts)/1000000))   
print('cost of GS:                     ' + str(ground_station.part_costs/1000000))
print('cost of salaries & contractors: ' + str(cost_salaries/1000000))
print('cost of manufacturing/Assembly: ' + str(sum(cost_of_manufacturing)/1000000))
print('-------------------------------------')
print('total costs:         ' + str(sum(cost_of_manufacturing+rd_cost+cost_of_launches+cost_of_transport+cost_of_parts+[ground_station.part_costs]+[cost_salaries])/1000000))



# Data to plot
plt.figure()
labels = 'Development', 'Transport', 'Purchases', 'Manufacturing/Assembly'

intertot = sum(cost_of_manufacturing+rd_cost+cost_of_transport+cost_of_parts)
sizes = [chartify(sum(rd_cost),intertot), chartify(sum(cost_of_transport),intertot), chartify(sum(cost_of_parts),intertot), chartify(sum(cost_of_manufacturing),intertot)]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0, 0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False, startangle=140, )
plt.title('Pie chart showing costs of the System')
plt.axis('equal')
plt.show()


#Data to plot
plt.figure()
labels = 'launches', 'Groundstation Operations', 'System'

sizes = [chartify(sum(cost_of_launches),total_cost), chartify(ground_station.part_costs+cost_salaries, total_cost),chartify(sum(cost_of_manufacturing+rd_cost+cost_of_transport+cost_of_parts),total_cost)]
colors = ['yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0, 0)  # explode 1st slice

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=False, startangle=240, )
plt.title('Pie chart showing relative costs to system')
plt.axis('equal')
plt.show()



