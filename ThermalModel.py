# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:35:21 2021

@author: cbalje
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" Define Variables """
albedo_earth = 0.4 # Infrared albedo factor of Earth, dimensionless
Solar_intensity = 1400 # Solar intensity in W/m2 - range between 1360 and 1420 W/m2 (SMAD)
Earth_IR_intensity = 235 # Earth infrared radiation intensity - range between 218 and 244 W/m2 (SMAD)
s_b = 5.67051*10**(-8) # Stefan-Boltzmann constant
cp_alu = 0.91*1000 # Specific heat of aluminium in J/kgK
cp_cop = 385.0 # Specific heat of copper in J/kgK
cp_ag = 240.0 # Specific heat of silver in J/kgK
k_alu = 205.0 # Thermal conductivity of aluminium in W/m K
k_cop = 386.0 # Thermal conductivity of copper in W/m K
K_ag = 406.0 # Thermal conductivity of silver in W/m K
Cap_HRS = 7000 # Capacity of heat rejection system in W/kg

Dish_L_Dia = 2*482 # Diameter of large paraboloid in m
Dish_L_Depth = 92.38 # Depth of large paraboloid in m
Dish_S_Dia = 50 # Diameter of small paraboloid in m
Dish_S_Depth = 4.79 # Depth of small paraboloid in m
Relay_Dia = 226.86*2 # Diameter of relay mirror in m
Reflector_Dia = 37.91*2 # Diameter of the reflector "Sting" in m
Struc_Len = sum([726.037*4, 387.17*2, 413.67*2]) # Strut length across (/diameter) in m
Lens_Thickness = 0.2 # Lens thickness in m
e_mirr_S = 0.94 # Rigid mirror efficiency, dimensionless
e_mirr_L = 0.91 # Foil mirror efficiency, dimensionless
# e_lens = 0.99 # Lens efficiency, dimensionless

def paraboloid_Area(diameter,depth):
    r,d = diameter/2,depth
    return (np.pi*r/(6*d**2))*((r**2+4*d**2)**(3/2)-r**3)

Mass_Dish_S = 30509.10 #paraboloid_Area(Dish_S_Dia,Dish_S_Depth)*(15+30)
Mass_Dish_L = 8555862.789 #paraboloid_Area(Dish_L_Dia,Dish_L_Depth)*(0.15+30)
Mass_Struc = sum([7857.502880076774, 721572.6489172676, 2104886.5915771737]) #Struc_Len*1500
#Mass_Lens = 0.25*np.pi*Dish_S_Dia**2*Lens_Thickness*2200
Mass_Relay = 272956.95 #(0.25*np.pi*Relay_Dia**2)*(15+30)
Mass_Reflector = 45614.4 #(0.25*np.pi*Reflector_Dia**2)*(15+30)
Mass_HRS = 820.0

""" 3D Shape and projections """
df_model = pd.DataFrame(index=["Dish_S_Sun",                            "Dish_S_Mirror",                            "Dish_L_Mirror",                                        "Dish_L_Back",                              "Structure_Support",                    "Reflector_Mirror",                             "Reflector_Back",                   "Relay_Mirror",                                     "Relay_Back"], columns = ["Emissivity", "Absorbtivity", "Total_Area","Eff_Sun_Area","Earth_Area_0","Earth_Area_90","Earth_Area_180","Earth_Area_270"])
df_model["Emissivity"] =     [0.95, 0.05, 0.05, 0.3, 0.95, 0.05, 0.95, 0.05, 0.95] 
df_model["Absorbtivity"] =   [0.35, 1-e_mirr_S, 1-e_mirr_L, 0.2, 0.94, 1-e_mirr_S, 0.35, 1-e_mirr_S, 0.35]
#df_model["Emissivity"] =    [0.95, 0.05, 0.05, 0.95, 0.95, 0.05, 0.95, 0.05, 0.95] 
#df_model["Absorbtivity"] =  [0.35, 1-e_mirr_S, 1-e_mirr_L, 0.94, 0.94, 1-e_mirr_S, 0.35, 1-e_mirr_S, 0.35]
#df_model["Heat_Capacity"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
df_model["Total_Area"] =     [paraboloid_Area(Dish_S_Dia,Dish_S_Depth), paraboloid_Area(Dish_S_Dia,Dish_S_Depth),   paraboloid_Area(Dish_L_Dia,Dish_L_Depth),               paraboloid_Area(Dish_L_Dia,Dish_L_Depth),   Struc_Len,                              3040.96,                                        3040.96,                            18197.13,                                           18197.13]
df_model["Eff_Sun_Area"] =   [(0.25*np.pi*Dish_S_Dia**2),               (0.25*np.pi*Dish_L_Dia**2)*e_mirr_L,        (0.25*np.pi*Dish_L_Dia**2),                             (0.0),                                      Struc_Len/8,                            (0.25*np.pi*Dish_L_Dia**2)*e_mirr_S*e_mirr_L,   (0.0),                              (0.25*np.pi*Dish_L_Dia**2)*e_mirr_S**2*e_mirr_L,    (0.0)]
df_model["Earth_Area_0"] =   [(0.0),                                    (0.0),                                      (0.0),                                                  (0.25*np.pi*Dish_L_Dia**2),                 (0.0),                                  (0.0),                                          (np.cos(np.radians(10))*3040.96),   (np.cos(np.radians(1))*18197.13),                   (0.0)]
df_model["Earth_Area_90"] =  [(4/3*Dish_S_Dia/2*Dish_S_Depth),          (0.0),                                      (0.0),                                                  (4/3*Dish_L_Dia/2*Dish_L_Depth),            (np.cos(np.radians(30))*Struc_Len/8),   (np.cos(np.radians(45))*3040.96),               (0.0),                              (np.cos(np.radians(80))*18197.13),                  (0.0)]
df_model["Earth_Area_180"] = [(0.25*np.pi*Dish_S_Dia**2),               (0.0),                                      ((0.25*np.pi*Dish_L_Dia**2)-(0.25*np.pi*Dish_S_Dia**2)),(0.0),                                      (Struc_Len/8),                          (0.0),                                          (0.0),                              (np.cos(np.radians(45))*18197.13),                  (0.0)]
df_model["Earth_Area_270"] = [( 4/3*Dish_S_Dia/2*Dish_S_Depth),         (0.0),                                      (0.0),                                                  (4/3*Dish_L_Dia/2*Dish_L_Depth),            (np.cos(np.radians(30))*Struc_Len/8),   (np.cos(np.radians(45))*3040.96),               (0.0),                              (np.cos(np.radians(80))*18197.13),                  (0.0)]

def Proj_Area_Earth(angle, diameter=Dish_L_Dia, depth=Dish_L_Depth, A_proj=[None, None, None, None]): # Projected area of a paraboloid with specified dimensions at a specified angle wrt Earth
    if A_proj[0] == None:
        A_proj[0]= (0.25*np.pi*diameter**2)
    if A_proj[1] == None:
        A_proj[1]= (4/3*diameter/2*depth)
    if A_proj[2] == None:
        A_proj[2] = A_proj[0]
    if A_proj[3] == None:
        A_proj[3] = A_proj[1] 
    if 0<=angle<=90:
        return A_proj[0]+angle/90*(A_proj[1]-A_proj[0])
    elif 90<angle<=180:
        return A_proj[1]+(angle-90)/90*(A_proj[2]-A_proj[1])
    elif 180<angle<=270:
        return A_proj[2]+(angle-180)/90*(A_proj[3]-A_proj[2])
    elif 270<angle<=360:
        return A_proj[3]+(angle-270)/90*(A_proj[0]-A_proj[3])
    else:
        return None


def func_Qrad(em,Atot,Temp):
    return em*s_b*Atot*Temp**4

def func_qHRS(tandem, T, HRS_Mass=Mass_HRS):
    return -tandem*HRS_Mass*Cap_HRS*min((T**4/(550**4)),1)
    
def func_qsol(ab, Js, Aeff):
    return ab*Js*Aeff

def func_qalb(ab, Js, albedo, Aeff):
    return ab*Js*albedo*Aeff

def func_qIR(em, JIR, Aeff):
    return em*JIR*Aeff

def func_Qext(ab,em,albedo_eff,Js,JIR,Aeffsol,Aeffearth):
    qsol = func_qsol(ab, Js, Aeffsol)
    qalb = func_qalb(ab, Js, albedo_eff, Aeffearth)
    qIR = func_qIR(em, JIR, Aeffearth)
    return qsol+qalb+qIR

def func_Qint_from1to2(T1,T2,Aint, k=k_alu):
    return -k*Aint*(T2 - T1)

""" Set-up Time and Orbit """
dt = 10 # in seconds
hours = 72
df_orbit = pd.DataFrame(np.arange(0,hours*3600,dt), columns=["Timestep"])
df_orbit["Orbit_Angle"] = (df_orbit["Timestep"]/86400 * 360)%360
df_orbit["albedo_eff"] = np.where(df_orbit["Orbit_Angle"] < 90.0, np.cos(np.radians(df_orbit["Orbit_Angle"]))*albedo_earth, 0 )
df_orbit["albedo_eff"] = np.where((df_orbit["Orbit_Angle"]) > 270.0, np.cos(np.radians(df_orbit["Orbit_Angle"]))*albedo_earth, df_orbit["albedo_eff"])

df_T = pd.DataFrame(columns=['Dish_S_Sun','Dish_S_Mirror', 'Dish_L_Mirror', 'Dish_L_Back', 'Structure_Support', "Reflector_Mirror", "Reflector_Back", 'Relay_Mirror', 'Relay_Back']) # Initial internal temperature
df_T.loc[0] = 20 + 273.15
df_cp_alu = pd.DataFrame(columns=['Dish_S_Sun','Dish_S_Mirror', 'Dish_L_Mirror', 'Dish_L_Back', 'Structure_Support', "Reflector_Mirror", "Reflector_Back", 'Relay_Mirror', 'Relay_Back']) # Initial 
df_cp_alu.loc[0] = 4186.798188*np.exp((-3.3767 + 2.4552*(np.log(df_T.loc[0])-np.log(50.2698)) - 1.1284*(np.log(df_T.loc[0])-np.log(50.2698))**2 + 0.18572*(np.log(df_T.loc[0])-np.log(50.2698))**3))

""" Generate Data """

for component in df_model.index:
    df_orbit[component+"_qsol"] = 0
    df_orbit[component+"_qalb"] = 0
    df_orbit[component+"_qIR"] = 0
    df_orbit[component+"_Qext"] = 0
    df_orbit[component+"_Qrad"] = 0
    df_orbit[component+"_heatbalance"] = 0


for i in df_orbit.index:
    if (i+1)%(len(df_orbit.index)/10-len(df_orbit.index)/10%1)==0:
        print((i+1)/len(df_orbit.index)*100, " %", "  -  T_avg = ", df_T.loc[i].mean()-273.15, " degC")
    for component in df_model.index:
        df_cp_alu.at[i, component] = 4186.798188*np.exp((-3.3767 + 2.4552*(np.log(df_T.loc[i][component])-np.log(50.2698)) - 1.1284*(np.log(df_T.loc[i][component])-np.log(50.2698))**2 + 0.18572*(np.log(df_T.loc[i][component])-np.log(50.2698))**3))
        ab,em = df_model.loc[component]["Absorbtivity"], df_model.loc[component]["Emissivity"]
        Aeffsol, Atot = df_model.loc[component]["Eff_Sun_Area"], df_model.loc[component]["Total_Area"]
        A_Earth_0, A_Earth_90, A_Earth_180, A_Earth_270 = df_model.loc[component]["Earth_Area_0"], df_model.loc[component]["Earth_Area_90"], df_model.loc[component]["Earth_Area_180"], df_model.loc[component]["Earth_Area_270"]
        df_orbit.at[i, component+"_qsol"] = func_qsol(ab, Solar_intensity, Aeffsol)
        df_orbit.at[i, component+"_qalb"] = func_qalb(ab, Solar_intensity, df_orbit["albedo_eff"].iloc[i], Proj_Area_Earth(df_orbit["Orbit_Angle"].iloc[i], A_proj=[A_Earth_0, A_Earth_90, A_Earth_180, A_Earth_270]))
        df_orbit.at[i, component+"_qIR"] = func_qIR(em, Earth_IR_intensity, Proj_Area_Earth(df_orbit["Orbit_Angle"].iloc[i], A_proj=[A_Earth_0, A_Earth_90, A_Earth_180, A_Earth_270]))
        df_orbit.at[i, component+"_Qext"] = df_orbit[component+"_qsol"].iloc[i]+df_orbit[component+"_qalb"].iloc[i]+df_orbit[component+"_qIR"].iloc[i]
        df_orbit.at[i, component+"_Qrad"] = func_Qrad(em, Atot, df_T.loc[i][component])
        df_orbit.at[i, component+"_heatbalance"] = df_orbit[component+"_Qext"].iloc[i]-df_orbit[component+"_Qrad"].iloc[i]
    
    df_T.loc[i+1] = df_T.loc[i]
    df_T.loc[i+1]['Dish_S_Sun'] += (func_qHRS(21, df_T.loc[i]['Dish_S_Sun']) + df_orbit["Dish_S_Sun_heatbalance"].iloc[i]+df_orbit["Dish_S_Mirror_heatbalance"].iloc[i]+func_Qint_from1to2(df_T.loc[i]['Structure_Support'], df_T.loc[i]['Dish_S_Sun'], paraboloid_Area(Dish_S_Dia,Dish_S_Depth)/8))/(df_cp_alu.loc[i]['Dish_S_Sun']*Mass_Dish_S)*dt
    df_T.loc[i+1]['Dish_S_Mirror'] = df_T.loc[i+1]['Dish_S_Sun']
    df_T.loc[i+1]['Dish_L_Mirror'] += (df_orbit["Dish_L_Back_heatbalance"].iloc[i]+df_orbit["Dish_L_Mirror_heatbalance"].iloc[i]+func_Qint_from1to2(df_T.loc[i]['Structure_Support'], df_T.loc[i]['Dish_L_Mirror'], Struc_Len/2))/(df_cp_alu.loc[i]['Dish_L_Mirror']*Mass_Dish_L)*dt
    df_T.loc[i+1]['Dish_L_Back'] = df_T.loc[i+1]['Dish_L_Mirror']
    df_T.loc[i+1]['Structure_Support'] += (df_orbit["Structure_Support_heatbalance"].iloc[i] + func_Qint_from1to2(df_T.loc[i]['Relay_Back'], df_T.loc[i]['Structure_Support'], (0.25*np.pi*Reflector_Dia**2)/8) + func_Qint_from1to2(df_T.loc[i]['Reflector_Mirror'], df_T.loc[i]['Structure_Support'], (0.25*np.pi*Reflector_Dia**2)/8) + func_Qint_from1to2(df_T.loc[i]['Dish_S_Sun'], df_T.loc[i]['Structure_Support'], paraboloid_Area(Dish_S_Dia,Dish_S_Depth)/8) + func_Qint_from1to2(df_T.loc[i]['Dish_L_Mirror'], df_T.loc[i]['Structure_Support'], Struc_Len/2))/(df_cp_alu.loc[i]['Structure_Support']*Mass_Struc)*dt
#    df_T.loc[i+1]['Lens'] += (df_orbit["Lens_heatbalance"].iloc[i] + func_Qint_from1to2(df_T.loc[i]['Dish_L_Mirror'], df_T.loc[i]['Lens'], (np.pi*Dish_S_Dia*Lens_Thickness)))/(df_cp_alu.loc[i][component]*Mass_Lens)*dt
    df_T.loc[i+1]['Reflector_Mirror'] += (func_qHRS(18, df_T.loc[i]['Reflector_Mirror'])+df_orbit["Reflector_Mirror_heatbalance"].iloc[i]+df_orbit["Reflector_Back_heatbalance"].iloc[i]+func_Qint_from1to2(df_T.loc[i]['Structure_Support'], df_T.loc[i]['Reflector_Mirror'], (0.25*np.pi*Reflector_Dia**2)/8))/(df_cp_alu.loc[i]['Reflector_Mirror']*Mass_Reflector)*dt
    df_T.loc[i+1]['Reflector_Back'] = df_T.loc[i+1]['Reflector_Mirror']
    df_T.loc[i+1]['Relay_Mirror'] += (df_orbit["Relay_Mirror_heatbalance"].iloc[i]+df_orbit["Relay_Back_heatbalance"].iloc[i]+func_Qint_from1to2(df_T.loc[i]['Structure_Support'], df_T.loc[i]['Relay_Back'], (0.25*np.pi*Reflector_Dia**2)/8))/(df_cp_alu.loc[i]['Relay_Mirror']*Mass_Relay)*dt
    df_T.loc[i+1]['Relay_Back'] = df_T.loc[i+1]['Relay_Mirror']
    last_entry = i

df_dT = df_T[:-1].copy()
for i in range(len(df_T)-1):
    df_dT.loc[i] = df_T.iloc[i+1]-df_T.iloc[i]    
df_dT = df_dT[50:]

""" Display Data """
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

def Display_Data(multx = 1.01, addy = 0.1, data = (df_T[["Dish_S_Mirror","Dish_L_Mirror","Structure_Support","Reflector_Mirror","Relay_Mirror"]]-273.15)):
    
    # These are the "Tableau 20" colors as RGB.    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
      
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)    
      
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    # Common sizes: (10, 7.5) and (12, 9)    
    plt.figure(figsize=(12, 9))    
      
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Temperature [degC]', fontsize=14) 
    
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    
      
    # Limit the range of the plot to only where the data is.    
    # Avoid unnecessary whitespace.    
    plt.ylim((min(df_T.min())*0.9-273.15), (max(df_T.max())*1.1-273.15))    
    plt.xlim(0, hours*3600)    
    
    # Make sure your axis ticks are large enough to be easily read.    
    # You don't want your viewers squinting to read your plot.    
    plt.yticks(fontsize=14)    
    plt.xticks(fontsize=14)    
    
    color = 1
    for column in data.columns:
        plt.plot((data[column].index*dt).tolist(), data[column].tolist(), lw=2.5, color=tableau20[color])
        y_pos = data[column].tolist()[-1]      
        # Again, make sure that all labels are large enough to be easily read    
        # by the viewer.    
        plt.text((data[column].index*dt).tolist()[-1]*multx, y_pos+addy, concatenate_list_data(column.split("_")[:-1]), fontsize=14, color=tableau20[color])   
        color+=1

Display_Data()



'''
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:35:21 2021

@author: cbalje
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" Define Variables """
albedo_earth = 0.4 # Infrared albedo factor of Earth, dimensionless
Solar_intensity = 1400 # Solar intensity in W/m2 - range between 1360 and 1420 W/m2 (SMAD)
Earth_IR_intensity = 235 # Earth infrared radiation intensity - range between 218 and 244 W/m2 (SMAD)
s_b = 5.67051*10**(-8) # Stefan-Boltzmann constant
cp_alu = 0.91*1000 # Specific heat of aluminium in J/kgK
cp_cop = 385.0 # Specific heat of copper in J/kgK
cp_ag = 240.0 # Specific heat of silver in J/kgK
k_alu = 205.0 # Thermal conductivity of aluminium in W/m K
k_cop = 386.0 # Thermal conductivity of copper in W/m K
K_ag = 406.0 # Thermal conductivity of silver in W/m K
Cap_HRS = 7000 # Capacity of heat rejection system in W/kg

Dish_L_Dia = 2*435 # Diameter of large paraboloid in m
Dish_L_Depth = 70.32 # Depth of large paraboloid in m
Dish_S_Dia = 50 # Diameter of small paraboloid in m
Dish_S_Depth = 4.04 # Depth of small paraboloid in m
Relay_Dia = 174.47*2 # Diameter of relay mirror in m
Reflector_Dia = 37.1*2 # Diameter of the reflector "Sting" in m
Struc_Len = 746.83*3+320.36*2+443.9 # Strut length across (/diameter) in m
Lens_Thickness = 0.2 # Lens thickness in m
e_mirr_S = 0.94 # Rigid mirror efficiency, dimensionless
e_mirr_L = 0.91 # Foil mirror efficiency, dimensionless
# e_lens = 0.99 # Lens efficiency, dimensionless

def paraboloid_Area(diameter,depth):
    r,d = diameter/2,depth
    return (np.pi*r/(6*d**2))*((r**2+4*d**2)**(3/2)-r**3)

Mass_Dish_S = 30509.10 #paraboloid_Area(Dish_S_Dia,Dish_S_Depth)*(15+30)
Mass_Dish_L = 8555862.789 #paraboloid_Area(Dish_L_Dia,Dish_L_Depth)*(0.15+30)
Mass_Struc = sum([6106.561944397918, 721626.5252575839, 2104886.5915771737]) #Struc_Len*1500
#Mass_Lens = 0.25*np.pi*Dish_S_Dia**2*Lens_Thickness*2200
Mass_Relay = 272956.95 #(0.25*np.pi*Relay_Dia**2)*(15+30)
Mass_Reflector = 45614.4 #(0.25*np.pi*Reflector_Dia**2)*(15+30)
Mass_HRS = 0.0

""" 3D Shape and projections """
df_model = pd.DataFrame(index=["Dish_S_Sun", "Dish_S_Mirror", "Dish_L_Mirror", "Dish_L_Back", "Structure_Support", "Reflector_Mirror", "Reflector_Back", "Relay_Mirror", "Relay_Back"], columns = ["Emissivity", "Absorbtivity", "Total_Area","Eff_Sun_Area","Earth_Area_0","Earth_Area_90","Earth_Area_180","Earth_Area_270"])
#df_model["Emissivity"] =  [0.95, 0.05, 0.05, 0.3, 0.95, 0.05, 0.95, 0.05, 0.95] 
#df_model["Absorbtivity"] = [0.35, 1-e_mirr_S, 1-e_mirr_L, 0.2, 0.94, 1-e_mirr_S, 0.35, 1-e_mirr_S, 0.35]
df_model["Emissivity"] =  [0.95, 0.05, 0.05, 0.95, 0.95, 0.05, 0.95, 0.05, 0.95] 
df_model["Absorbtivity"] = [0.35, 1-e_mirr_S, 1-e_mirr_L, 0.94, 0.94, 1-e_mirr_S, 0.35, 1-e_mirr_S, 0.35]
#df_model["Heat_Capacity"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
df_model["Total_Area"] = [paraboloid_Area(Dish_S_Dia,Dish_S_Depth), paraboloid_Area(Dish_S_Dia,Dish_S_Depth), paraboloid_Area(Dish_L_Dia,Dish_L_Depth), paraboloid_Area(Dish_L_Dia,Dish_L_Depth), Struc_Len, (0.25*np.pi*Reflector_Dia**2), (0.25*np.pi*Reflector_Dia**2), (0.25*np.pi*Relay_Dia**2), (0.25*np.pi*Relay_Dia**2)]
df_model["Eff_Sun_Area"] = [(0.25*np.pi*Dish_S_Dia**2), (0.25*np.pi*Dish_L_Dia**2)*e_mirr_L, (0.25*np.pi*Dish_L_Dia**2), (0.0), Struc_Len, (0.25*np.pi*Dish_L_Dia**2)*e_mirr_S*e_mirr_L, (0.0), (0.25*np.pi*Dish_L_Dia**2)*e_mirr_S**2*e_mirr_L, (0.0)]
df_model["Earth_Area_0"] = [(0.0), (0.0), (0.0), (0.25*np.pi*Dish_L_Dia**2), (0.0), (0.0), (np.cos(np.radians(0))*0.25*np.pi*Reflector_Dia**2), (np.cos(np.radians(0))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_90"] = [(4/3*Dish_S_Dia/2*Dish_S_Depth), (0.0), (0.0), (4/3*Dish_L_Dia/2*Dish_L_Depth), (np.cos(np.radians(30))*Struc_Len), (0.0), (np.cos(np.radians(0))*0.25*np.pi*Reflector_Dia**2), (np.cos(np.radians(45))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_180"] = [(0.25*np.pi*Dish_S_Dia**2), (0.0), ((0.25*np.pi*Dish_L_Dia**2)-(0.25*np.pi*Dish_S_Dia**2)), (0.0), (Struc_Len), (0.0), (np.cos(np.radians(0))*0.25*np.pi*Reflector_Dia**2), (np.cos(np.radians(0))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_270"] = [( 4/3*Dish_S_Dia/2*Dish_S_Depth), (0.0), (0.0), (4/3*Dish_L_Dia/2*Dish_L_Depth), (np.cos(np.radians(30))*Struc_Len), (0.0), (np.cos(np.radians(0))*0.25*np.pi*Reflector_Dia**2), (np.cos(np.radians(45))*0.25*np.pi*Relay_Dia**2), (0.0)]

def Proj_Area_Earth(angle, diameter=Dish_L_Dia, depth=Dish_L_Depth, A_proj=[None, None, None, None]): # Projected area of a paraboloid with specified dimensions at a specified angle wrt Earth
    if A_proj[0] == None:
        A_proj[0]= (0.25*np.pi*diameter**2)
    if A_proj[1] == None:
        A_proj[1]= (4/3*diameter/2*depth)
    if A_proj[2] == None:
        A_proj[2] = A_proj[0]
    if A_proj[3] == None:
        A_proj[3] = A_proj[1] 
    if 0<=angle<=90:
        return A_proj[0]+angle/90*(A_proj[1]-A_proj[0])
    elif 90<angle<=180:
        return A_proj[1]+(angle-90)/90*(A_proj[2]-A_proj[1])
    elif 180<angle<=270:
        return A_proj[2]+(angle-180)/90*(A_proj[3]-A_proj[2])
    elif 270<angle<=360:
        return A_proj[3]+(angle-270)/90*(A_proj[0]-A_proj[3])
    else:
        return None


def func_Qrad(em,Atot,Temp):
    return em*s_b*Atot*Temp**4

def func_qHRS(T, HRS_Mass=Mass_HRS):
    return -HRS_Mass*Cap_HRS*min((T**4/(550**4)),1)
    
def func_qsol(ab, Js, Aeff):
    return ab*Js*Aeff

def func_qalb(ab, Js, albedo, Aeff):
    return ab*Js*albedo*Aeff

def func_qIR(em, JIR, Aeff):
    return em*JIR*Aeff

def func_Qext(ab,em,albedo_eff,Js,JIR,Aeffsol,Aeffearth):
    qsol = func_qsol(ab, Js, Aeffsol)
    qalb = func_qalb(ab, Js, albedo_eff, Aeffearth)
    qIR = func_qIR(em, JIR, Aeffearth)
    return qsol+qalb+qIR

def func_Qint_from1to2(T1,T2,Aint, k=k_alu):
    return -k*Aint*(T2 - T1)

""" Set-up Time and Orbit """
dt = 10 # in seconds
hours = 12
df_orbit = pd.DataFrame(np.arange(0,hours*3600,dt), columns=["Timestep"])
df_orbit["Orbit_Angle"] = (df_orbit["Timestep"]/86400 * 360)%360
df_orbit["albedo_eff"] = np.where(df_orbit["Orbit_Angle"] < 90.0, np.cos(np.radians(df_orbit["Orbit_Angle"]))*albedo_earth, 0 )
df_orbit["albedo_eff"] = np.where((df_orbit["Orbit_Angle"]) > 270.0, np.cos(np.radians(df_orbit["Orbit_Angle"]))*albedo_earth, df_orbit["albedo_eff"])

df_T = pd.DataFrame(columns=['Dish_S_Sun','Dish_S_Mirror', 'Dish_L_Mirror', 'Dish_L_Back', 'Structure_Support', "Reflector_Mirror", "Reflector_Back", 'Relay_Mirror', 'Relay_Back']) # Initial internal temperature
df_T.loc[0] = 20 + 273.15
df_cp_alu = pd.DataFrame(columns=['Dish_S_Sun','Dish_S_Mirror', 'Dish_L_Mirror', 'Dish_L_Back', 'Structure_Support', "Reflector_Mirror", "Reflector_Back", 'Relay_Mirror', 'Relay_Back']) # Initial 
df_cp_alu.loc[0] = 4186.798188*np.exp((-3.3767 + 2.4552*(np.log(df_T.loc[0])-np.log(50.2698)) - 1.1284*(np.log(df_T.loc[0])-np.log(50.2698))**2 + 0.18572*(np.log(df_T.loc[0])-np.log(50.2698))**3))

""" Generate Data """

for component in df_model.index:
    df_orbit[component+"_qsol"] = 0
    df_orbit[component+"_qalb"] = 0
    df_orbit[component+"_qIR"] = 0
    df_orbit[component+"_Qext"] = 0
    df_orbit[component+"_Qrad"] = 0
    df_orbit[component+"_heatbalance"] = 0


for i in df_orbit.index:
    if (i+1)%(len(df_orbit.index)/10-len(df_orbit.index)/10%1)==0:
        print((i+1)/len(df_orbit.index)*100, " %", "  -  T_avg = ", df_T.loc[i].mean()-273.15, " degC")
    for component in df_model.index:
        df_cp_alu.at[i, component] = 4186.798188*np.exp((-3.3767 + 2.4552*(np.log(df_T.loc[i][component])-np.log(50.2698)) - 1.1284*(np.log(df_T.loc[i][component])-np.log(50.2698))**2 + 0.18572*(np.log(df_T.loc[i][component])-np.log(50.2698))**3))
        ab,em = df_model.loc[component]["Absorbtivity"], df_model.loc[component]["Emissivity"]
        Aeffsol, Atot = df_model.loc[component]["Eff_Sun_Area"], df_model.loc[component]["Total_Area"]
        A_Earth_0, A_Earth_90, A_Earth_180, A_Earth_270 = df_model.loc[component]["Earth_Area_0"], df_model.loc[component]["Earth_Area_90"], df_model.loc[component]["Earth_Area_180"], df_model.loc[component]["Earth_Area_270"]
        df_orbit.at[i, component+"_qsol"] = func_qsol(ab, Solar_intensity, Aeffsol)
        df_orbit.at[i, component+"_qalb"] = func_qalb(ab, Solar_intensity, df_orbit["albedo_eff"].iloc[i], Proj_Area_Earth(df_orbit["Orbit_Angle"].iloc[i], A_proj=[A_Earth_0, A_Earth_90, A_Earth_180, A_Earth_270]))
        df_orbit.at[i, component+"_qIR"] = func_qIR(em, Earth_IR_intensity, Proj_Area_Earth(df_orbit["Orbit_Angle"].iloc[i], A_proj=[A_Earth_0, A_Earth_90, A_Earth_180, A_Earth_270]))
        df_orbit.at[i, component+"_Qext"] = df_orbit[component+"_qsol"].iloc[i]+df_orbit[component+"_qalb"].iloc[i]+df_orbit[component+"_qIR"].iloc[i]
        df_orbit.at[i, component+"_Qrad"] = func_Qrad(em, Atot, df_T.loc[i][component])
        df_orbit.at[i, component+"_heatbalance"] = df_orbit[component+"_Qext"].iloc[i]-df_orbit[component+"_Qrad"].iloc[i]
    
    df_T.loc[i+1] = df_T.loc[i]
    df_T.loc[i+1]['Dish_S_Sun'] += (func_qHRS(df_T.loc[i]['Dish_S_Sun']) + df_orbit["Dish_S_Sun_heatbalance"].iloc[i]+df_orbit["Dish_S_Mirror_heatbalance"].iloc[i]+func_Qint_from1to2(df_T.loc[i]['Structure_Support'], df_T.loc[i]['Dish_S_Sun'], paraboloid_Area(Dish_S_Dia,Dish_S_Depth)/2))/(df_cp_alu.loc[i]['Dish_S_Sun']*Mass_Dish_S)*dt
    df_T.loc[i+1]['Dish_S_Mirror'] = df_T.loc[i+1]['Dish_S_Sun']
    df_T.loc[i+1]['Dish_L_Mirror'] += (df_orbit["Dish_L_Back_heatbalance"].iloc[i]+df_orbit["Dish_L_Mirror_heatbalance"].iloc[i]+func_Qint_from1to2(df_T.loc[i]['Structure_Support'], df_T.loc[i]['Dish_L_Mirror'], Struc_Len))/(df_cp_alu.loc[i]['Dish_L_Mirror']*Mass_Dish_L)*dt
    df_T.loc[i+1]['Dish_L_Back'] = df_T.loc[i+1]['Dish_L_Mirror']
    df_T.loc[i+1]['Structure_Support'] += (df_orbit["Structure_Support_heatbalance"].iloc[i] + func_Qint_from1to2(df_T.loc[i]['Reflector_Mirror'], df_T.loc[i]['Structure_Support'], (0.25*np.pi*Reflector_Dia**2)/4) + func_Qint_from1to2(df_T.loc[i]['Dish_S_Sun'], df_T.loc[i]['Structure_Support'], paraboloid_Area(Dish_S_Dia,Dish_S_Depth)) + func_Qint_from1to2(df_T.loc[i]['Dish_L_Mirror'], df_T.loc[i]['Structure_Support'], Struc_Len))/(df_cp_alu.loc[i]['Structure_Support']*Mass_Struc)*dt
#    df_T.loc[i+1]['Lens'] += (df_orbit["Lens_heatbalance"].iloc[i] + func_Qint_from1to2(df_T.loc[i]['Dish_L_Mirror'], df_T.loc[i]['Lens'], (np.pi*Dish_S_Dia*Lens_Thickness)))/(df_cp_alu.loc[i][component]*Mass_Lens)*dt
    df_T.loc[i+1]['Reflector_Mirror'] += (func_qHRS(df_T.loc[i]['Reflector_Mirror'])+df_orbit["Reflector_Mirror_heatbalance"].iloc[i]+df_orbit["Reflector_Back_heatbalance"].iloc[i]+func_Qint_from1to2(df_T.loc[i]['Structure_Support'], df_T.loc[i]['Reflector_Mirror'], (0.25*np.pi*Reflector_Dia**2)/4))/(df_cp_alu.loc[i]['Reflector_Mirror']*Mass_Reflector)*dt
    df_T.loc[i+1]['Reflector_Back'] = df_T.loc[i+1]['Reflector_Mirror']
    df_T.loc[i+1]['Relay_Mirror'] += (df_orbit["Relay_Mirror_heatbalance"].iloc[i]+df_orbit["Relay_Back_heatbalance"].iloc[i])/(df_cp_alu.loc[i]['Relay_Mirror']*Mass_Relay)*dt
    df_T.loc[i+1]['Relay_Back'] = df_T.loc[i+1]['Relay_Mirror']
    last_entry = i

    

""" Display Data """
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

def Display_Data(multx = 1.01, addy = 0.1, data = (df_T[["Dish_S_Mirror","Dish_L_Mirror","Structure_Support","Reflector_Mirror","Relay_Mirror"]]-273.15)):
    
    # These are the "Tableau 20" colors as RGB.    
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
      
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
    for i in range(len(tableau20)):    
        r, g, b = tableau20[i]    
        tableau20[i] = (r / 255., g / 255., b / 255.)    
      
    # You typically want your plot to be ~1.33x wider than tall. This plot is a rare    
    # exception because of the number of lines being plotted on it.    
    # Common sizes: (10, 7.5) and (12, 9)    
    plt.figure(figsize=(12, 9))    
      
    # Remove the plot frame lines. They are unnecessary chartjunk.    
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Temperature [degC]', fontsize=14) 
    
    # Ensure that the axis ticks only show up on the bottom and left of the plot.    
    # Ticks on the right and top of the plot are generally unnecessary chartjunk.    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()    
      
    # Limit the range of the plot to only where the data is.    
    # Avoid unnecessary whitespace.    
    plt.ylim((min(df_T.min())*0.9-273.15), (max(df_T.max())*1.1-273.15))    
    plt.xlim(0, hours*3600)    
    
    # Make sure your axis ticks are large enough to be easily read.    
    # You don't want your viewers squinting to read your plot.    
    plt.yticks(fontsize=14)    
    plt.xticks(fontsize=14)    
    
    color = 1
    for column in data.columns:
        plt.plot((data[column].index*dt).tolist(), data[column].tolist(), lw=2.5, color=tableau20[color])
        y_pos = data[column].tolist()[-1]      
        # Again, make sure that all labels are large enough to be easily read    
        # by the viewer.    
        plt.text((data[column].index*dt).tolist()[-1]*multx, y_pos+addy, concatenate_list_data(column.split("_")[:-1]), fontsize=14, color=tableau20[color])   
        color+=1

Display_Data()
''' 