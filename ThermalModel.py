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


Dish_L_Dia = 500 # Diameter of large paraboloid in m
Dish_L_Depth = 125 # Depth of large paraboloid in m
Dish_S_Dia = 20 # Diameter of small paraboloid in m
Dish_S_Depth = 2.5 # Depth of small paraboloid in m
Relay_Dia = 126.421758*2 # Diameter of relay mirror in m
e_mirr = 0.98 # Mirror efficiency, dimensionless
e_lens = 0.99 # Lens efficiency, dimensionless

def paraboloid_Area(diameter,depth):
    r,d = diameter/2,depth
    return (np.pi*r/(6*d**2))*((r**2+4*d**2)**(3/2)-r**3)

Mass_Dish_S = paraboloid_Area(Dish_S_Dia,Dish_S_Depth)*15
Mass_Dish_L = paraboloid_Area(Dish_L_Dia,Dish_L_Depth)*0.15
Mass_Struc = 4*Dish_L_Dia*1500
Mass_Lens = 0.25*np.pi*Dish_S_Dia**2*0.2*2200
Mass_Relay = (0.25*np.pi*Relay_Dia**2)*15

""" 3D Shape and projections """
df_model = pd.DataFrame(index=["Dish_S_Sun", "Dish_S_Mirror", "Dish_L_Mirror", "Dish_L_Back", "Add_Struc", "Lens", "Relay_Mirror", "Relay_Back"], columns = ["Emissivity", "Absorbtivity", "Heat_Capacity", "Total_Area","Eff_Sun_Area","Earth_Area_0","Earth_Area_90","Earth_Area_180","Earth_Area_270"])
df_model["Emissivity"] =  [0.95, 0.05, 0.05, 0.95, 0.95, 0.93, 0.05, 0.95] 
df_model["Absorbtivity"] = [0.35, 0.4, 0.4, 0.35, 0.35, 0.3, 0.4, 0.35]
#df_model["Heat_Capacity"] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
df_model["Total_Area"] = [paraboloid_Area(Dish_S_Dia,Dish_S_Depth), paraboloid_Area(Dish_S_Dia,Dish_S_Depth), paraboloid_Area(Dish_L_Dia,Dish_L_Depth), paraboloid_Area(Dish_L_Dia,Dish_L_Depth), 4*4*Dish_L_Dia, 2*Dish_S_Dia, (0.25*np.pi*Relay_Dia**2), (0.25*np.pi*Relay_Dia**2)]
df_model["Eff_Sun_Area"] = [(0.25*np.pi*Dish_S_Dia**2), (0.25*np.pi*Dish_L_Dia**2)*e_mirr, (0.25*np.pi*Dish_L_Dia**2), (0.0), 4*Dish_L_Dia, (0.25*np.pi*Dish_L_Dia**2)*e_mirr*e_mirr, (0.25*np.pi*Dish_L_Dia**2)*e_mirr*e_mirr*e_lens, (0.0)]
df_model["Earth_Area_0"] = [(0.0), (0.0), (0.0), (0.25*np.pi*Dish_L_Dia**2), (0.0), (0.0), (np.cos(np.radians(0))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_90"] = [(4/3*Dish_S_Dia/2*Dish_S_Depth), (0.0), (0.0), (4/3*Dish_L_Dia/2*Dish_L_Depth), (np.cos(np.radians(30))*2*Dish_L_Dia), (0.0), (np.cos(np.radians(45))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_180"] = [(0.25*np.pi*Dish_S_Dia**2), (0.0), ((0.25*np.pi*Dish_L_Dia**2)-(0.25*np.pi*Dish_S_Dia**2)), (0.0), (4*Dish_L_Dia), (0.0), (np.cos(np.radians(0))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_270"] = [(4/3*Dish_S_Dia/2*Dish_S_Depth), (0.0), (0.0), (4/3*Dish_L_Dia/2*Dish_L_Depth), (np.cos(np.radians(30))*2*Dish_L_Dia), (0.0), (np.cos(np.radians(45))*0.25*np.pi*Relay_Dia**2), (0.0)]

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


""" Set-up Time and Orbit """
dt = 10 # in seconds
hours = 72
df_orbit = pd.DataFrame(np.arange(0,hours*3600,dt), columns=["Timestep"])
df_orbit["Orbit_Angle"] = (df_orbit["Timestep"]/86400 * 360)%360
df_orbit["albedo_eff"] = np.where(df_orbit["Orbit_Angle"] < 90.0, np.cos(np.radians(df_orbit["Orbit_Angle"]))*albedo_earth, 0 )
df_orbit["albedo_eff"] = np.where((df_orbit["Orbit_Angle"]) > 270.0, np.cos(np.radians(df_orbit["Orbit_Angle"]))*albedo_earth, df_orbit["albedo_eff"])

df_T = pd.DataFrame(columns=['Dish_S_Sun','Dish_S_Mirror', 'Dish_L_Mirror', 'Dish_L_Back', 'Add_Struc', 'Lens', 'Relay_Mirror', 'Relay_Back']) # Initial internal temperature
df_T.loc[0] = 20 + 273.15
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
    df_T.loc[i+1]['Dish_S_Sun'] += (df_orbit["Dish_S_Sun_heatbalance"].iloc[i]+df_orbit["Dish_S_Mirror_heatbalance"].iloc[i])/(cp_alu*Mass_Dish_S)*dt
    df_T.loc[i+1]['Dish_S_Mirror'] = df_T.loc[i+1]['Dish_S_Sun']
    df_T.loc[i+1]['Dish_L_Mirror'] += (df_orbit["Dish_L_Back_heatbalance"].iloc[i]+df_orbit["Dish_L_Mirror_heatbalance"].iloc[i])/(cp_alu*Mass_Dish_L)*dt
    df_T.loc[i+1]['Dish_L_Back'] = df_T.loc[i+1]['Dish_L_Mirror']
    df_T.loc[i+1]['Add_Struc'] += (df_orbit["Add_Struc_heatbalance"].iloc[i])/(cp_alu*Mass_Struc)*dt
    df_T.loc[i+1]['Lens'] += (df_orbit["Lens_heatbalance"].iloc[i])/(cp_alu*Mass_Lens)*dt
    df_T.loc[i+1]['Relay_Mirror'] += (df_orbit["Relay_Mirror_heatbalance"].iloc[i]+df_orbit["Relay_Back_heatbalance"].iloc[i])/(cp_alu*Mass_Relay)*dt
    df_T.loc[i+1]['Relay_Back'] = df_T.loc[i+1]['Relay_Mirror']


""" Display Data """
df_T[["Dish_S_Mirror","Dish_L_Mirror","Add_Struc","Lens","Relay_Mirror"]].plot()

