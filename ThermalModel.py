# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 11:35:21 2021

@author: cbalje
"""
import pandas as pd
import numpy as np


""" Define Variables """
A_earth = 0.4 # Infrared albedo factor of Earth, dimensionless
Solar_intensity = 1400 # Solar intensity in W/m2

Dish_L_Dia = 500 # Diameter of large paraboloid in m
Dish_L_Depth = 125 # Depth of large paraboloid in m
Dish_S_Dia = 20 # Diameter of small paraboloid in m
Dish_S_Depth = 2.5 # Depth of small paraboloid in m
Relay_Dia = 126*2 # Diameter of relay mirror in m
e_mirr = 0.98 # Mirror efficiency, dimensionless
e_lens = 0.99 # Lens efficiency, dimensionless

def paraboloid_Area(diameter,depth):
    r,d = diameter/2,depth
    return (np.pi*r/(6*d**2))*((r**2+4*d**2)**(3/2)-r**3)

df_model = pd.DataFrame(index=["Dish_S_Sun", "Dish_S_Mirror", "Dish_L_Mirror", "Dish_L_Back", "Add_Struc", "Lens", "Relay_Mirror", "Relay_Back"], columns = ["Emissivity", "Reflectance", "Total_Area","Eff_Sun_Area","Earth_Area_0","Earth_Area_90","Earth_Area_180","Earth_Area_270"])
df_model["Emissivity"] =  [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2] 
df_model["Reflectance"] = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
df_model["Total_Area"] = [paraboloid_Area(Dish_S_Dia,Dish_S_Depth), paraboloid_Area(Dish_S_Dia,Dish_S_Depth), paraboloid_Area(Dish_L_Dia,Dish_L_Depth), paraboloid_Area(Dish_L_Dia,Dish_L_Depth), 4*4*Dish_L_Dia, 2*Dish_S_Dia, (0.25*np.pi*Relay_Dia**2), (0.25*np.pi*Relay_Dia**2)]
df_model["Eff_Sun_Area"] = [(0.25*np.pi*Dish_S_Dia**2), (0.25*np.pi*Dish_L_Dia**2)*e_mirr, (0.25*np.pi*Dish_L_Dia**2), (0.0), 4*Dish_L_Dia, (0.25*np.pi*Dish_L_Dia**2)*e_mirr*e_mirr, (0.25*np.pi*Dish_L_Dia**2)*e_mirr*e_mirr*e_lens, (0.0)]
df_model["Earth_Area_0"] = [(0.0), (0.0), (0.0), (0.25*np.pi*Dish_L_Dia**2), (0.0), (0.0), (np.cos(np.radians(0))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_90"] = [(4/3*Dish_S_Dia/2*Dish_S_Depth), (0.0), (0.0), (4/3*Dish_L_Dia/2*Dish_L_Depth), (np.cos(np.radians(30))*2*Dish_L_Dia), (0.0), (np.cos(np.radians(45))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_180"] = [(0.25*np.pi*Dish_S_Dia**2), (0.0), ((0.25*np.pi*Dish_L_Dia**2)-(0.25*np.pi*Dish_S_Dia**2)), (0.0), (4*Dish_L_Dia), (0.0), (np.cos(np.radians(0))*0.25*np.pi*Relay_Dia**2), (0.0)]
df_model["Earth_Area_270"] = [(4/3*Dish_S_Dia/2*Dish_S_Depth), (0.0), (0.0), (4/3*Dish_L_Dia/2*Dish_L_Depth), (np.cos(np.radians(30))*2*Dish_L_Dia), (0.0), (np.cos(np.radians(45))*0.25*np.pi*Relay_Dia**2), (0.0)]

""" Set-up Time and Orbit """
df_orbit = pd.DataFrame(np.arange(0,24*3600,10), columns=["Timestep"])
df_orbit["Orbit_Angle"] = df_orbit["Timestep"]/86400 * 360

df_orbit["A_earth_eff"] = np.where(df_orbit["Orbit_Angle"] < 90.0, np.cos(np.radians(df_orbit["Orbit_Angle"]))*A_earth, 0 )
df_orbit["A_earth_eff"] = np.where((df_orbit["Orbit_Angle"]) > 270.0, np.cos(np.radians(df_orbit["Orbit_Angle"]))*A_earth, df_orbit["A_earth_eff"])


""" 3D Shape and projections """
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
        

""" Generate Data """



""" Display Data """


