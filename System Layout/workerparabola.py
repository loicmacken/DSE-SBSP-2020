import numpy as np


#optimal heat dissipation capacity
p_diss = 300 #w/m2

#power calculation
p_ground = 100*10**6 #W
n_ground = 0.27
n_atmosphere = 0.55
n_rigid = 0.98
n_foil = 0.95
flux_solar = 1361 #W/m2

p_intake = p_ground/(n_ground*n_atmosphere*(n_rigid**2)*(n_foil**2))

s_intake = p_intake/flux_solar
r_intake = np.sqrt(s_intake)/np.pi

#thermal power @workerparabola
p_therm_worker = (p_intake*n_foil)*(1-n_foil) #worker parabola assumed to be foil for now, could be changed to rigid

s_rad =  p_therm_worker/p_diss
r_rad = np.sqrt(s_rad)/np.pi
print(r_rad)