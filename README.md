# DSE-SBSP-2020
Code repository for group 2 of the DSE in fall 2020 with the topic of Space Based Solar Power

Congratulations! You just wasted your time reading a stupid readme file!

# Data
All relevant data should be kept here, obviously. When you write a file in another directory you can use the *utils.py* function *get_data_root()* to get the path to this directory and access it. You will probably have to convert this path to a string for most import functions.

# Utils
This python file contains utility classes and functions that can be used for the project.
*get_project_root()* returns a path to the root of the project.
*get_data_root()* returns a path to the data directory.

## DataHandling
The DataHandling class is initiated simply with the path to the data directory, this path can be altered to save data to another location if desired.  

*import_centre_body()* provides a function to import central body data for a given planet or object from the *planetary_data.csv* file.  

*save_figure()* simply saves a matplotlib figure type as either a jpg or png depending on the filetype used (defaults to jpg). If a png is used, it defaults to 300dpi so the resolution is quite high (and hence only jpg is found in github). An alternative path can be passed to the function if you do not want to save the figure in a /figures directory within whichever data_path the class was initiated with.

## AstroUtils
This class stores methods that are specific to the astrodynamics and orbit propagation problem, as to keep it separate from other functions in the Utils.py file. Most of these methods are static and the class is not initiated with any variables, it is then subclassed by the OrbitPropagation class so that these methods can be easily used there (this structure may change at a later stage but it works for now so whatever).

# Astropy
## Orbital Propagation Class
An instance of this class requires the initial conditions of the orbit; position, velocity, time=0. It also requires data concerning the centre body, which in most cases for this project will be Earth.

This class defines three methods: *propagate_orbit, diffy_g, plot_3d*.

### Example
```
from src.utils import *  
cb_data = DataHandling().import_centre_body()

# Choose orbital body to get data from
data = cb_data['earth']

# ISS
# Altitude, Eccentricity, Inclination, True anomaly, Argument of Perigee, Right Acsension of Ascending Node

c0 = [data['radius'] + 414.0, 0.0000867, 51.6417, 0.0, 165.1519, 95.4907]

# Time span of two days
tspan = 3600 * 24 * 2

# Creat instances
o0 = OrbitPropagator(c0,
                     tspan=tspan,
                     dt=10.0,
                     coes=True,
                     cb=data,
                     perts=AstroUtils.init_perts(J2=True))

o0.propagate_orbit()

AstroUtils.plot_n_orbits([o0.rs],
                         labels=['ISS Orbit', 'Geostationary Orbit', 'IKAROS orbit'],
                         show_plot=True,
                         save_plot=True,
                         title=f'Orbit comparisons with J2 perturbation\n{round(o0.ts[-1][0] / 3600)} hrs')
```
<img src="data/figures/ISS_orbit_with_J2_perturbation.jpg" alt="orbital example" width="500px" height="500px">

## Plotting multiple orbits
You can plot multiple orbits using the *plot_n_orbits* function in *utils.py*.

### Example
```
from src.utils import *  
cb_data = DataHandling().import_centre_body()

# Choose orbital body to get data from
data = cb_data['earth']

# ISS
# Altitude, Eccentricity, Inclination, True anomaly, Argument of Perigee, Right Acsension of Ascending Node

c0 = [data['radius'] + 414.0, 0.0000867, 51.6417, 0.0, 165.1519, 95.4907]

# Geostationary
c1 = [data['radius'] + 35786, 0.0, 0.0, 0.0, 0.0, 0.0]

# IKAROS - LEO
c2 = [data['radius'] + 1750, 0.0, 0.0, 0.0, 0.0, 0.0]

tspan = 3600 * 24 * 2

# Creat instances
o0 = OrbitPropagator(c0,
                     tspan=tspan,
                     dt=10.0,
                     coes=True,
                     cb=data,
                     perts=AstroUtils.init_perts(J2=True))
o1 = OrbitPropagator(c1,
                     tspan=tspan,
                     dt=10.0,
                     coes=True,
                     cb=data,
                     perts=AstroUtils.init_perts(J2=True))
o2 = OrbitPropagator(c2,
                     tspan=tspan,
                     dt=10.0,
                     coes=True,
                     cb=data,
                     perts=AstroUtils.init_perts(J2=True))
o0.propagate_orbit()
o1.propagate_orbit()
o2.propagate_orbit() 

AstroUtils.plot_n_orbits([o0.rs, o1.rs, o2.rs],
                         labels=['ISS Orbit', 'Geostationary Orbit', 'IKAROS orbit'],
                         show_plot=True,
                         save_plot=True,
                         title=f'Orbit comparisons with J2 perturbation\n{round(o0.ts[-1][0] / 3600)} hrs')
```
<img src="data/figures/Orbit_comparisons_with_J2_perturbation.jpg" alt="orbital example" width="500px" height="500px">
