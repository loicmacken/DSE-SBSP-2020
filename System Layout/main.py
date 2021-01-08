from matplotlib import pyplot as plt
from parabola import *
import numpy as np

"""
NOTES

Important Nomenclature:
o_s =  sting offset
o_r = relay offset
r_s = sting radius
r_r = relay radius
A_s = sting area
A_r = relay Area
d_dish = dish depth
Parabola.length = length of curve segment of a parabola (Queen, worker)
FP = focal point
For the angles (Greek letters)(and general overview of blindspot mitigation) refer to sketch in Notion: subsystem outlines > downlink > blindspot mitigation draft

Body reference:
Worker to Queen offset is measured from rim to rim
sting offset is measured from Queen rim to center sting (reflector)
relay   "    "      "       "   "   "   "   "    relay
so the angles are measured consistent with the drawings in notion, but the distances are not (or not always)

Mass Estimations:
Reflectors (sting+relay): 1 area + 4 radial trusses + 1 circumferential truss + 2 connecting trusses to main struct
Queen+worker: 1 area + 4 "radial" trusses following the shape of the curve + 4 connecting trusses
all much like Chris's drawing

Note:
The code is kinda inefficient and takes around 20 secs, so be patient
alternatively you can change the number of steps in the iterations
"""

# Mass estimates [kg] (as in order of magnitude is in the correct range)
m_t = 1500  # mass of truss per m
m_m = 15  # mass of rigid mirror per m²
m_f = 0.15  # mass of reflective foil per m²
m_g = 75 # mass of mesh grid per m²

#------------
# USER INPUT
#------------
A_pv = 1.1*10**6 / (1362 * 0.27) # 5MW for bus power from pv cells
r_beam = 25
A_dish = (100*10**6 / (0.91 * 0.94**3 * 0.55 * 0.3)) / 1362 # 100 MW output from groundstation
#________________
#----------------

r_queen, w_d, w_pv = get_radius(r_beam, A_pv, A_dish)

print("Busy making", (r_queen//2)*750*1000*10, "different Honey configurations and calculating their mass so gimme a break OK\n")

DEPTHS = np.linspace(1, r_queen // 2 + 1, r_queen//2) # Range of big dish depths, upper limit relates to the depth where the FP would lie at the level of the dish rim

# setup ranges for mass minimisation
TOTAL_MASS = [] #this list will collect all the calculated total masses
RELAY_OFFSETS = [] # Queen rim to relay distance, corresponding to calculated mass
BEST_MARGIN = [] # gamma-beta overlap, corresponding to calculated mass

for d_queen in DEPTHS:
    d_worker = d_queen * r_beam / r_queen #small dish has same shape as big dish but scaled down, so same

    Queen = parabola(r_queen, d_queen)
    Worker = parabola(r_beam, d_worker, up=0)

    worker_offset = Queen.FP - Queen.d + Worker.FP - Worker.d #rim to rim distance between small and big dish
    struct1 = truss(-r_queen, -r_beam, 0, worker_offset)
    struct2 = truss(r_beam, r_queen, worker_offset, 0)

    RO = np.linspace(0, 750, 750) #range of relay offsets
    MASS = [] # setup lists for reflectors' (sting + relay) masses
    MARGINS = [] # setup list for beta-gamma overlap angles (see later)

    x = np.linspace(d_queen, d_queen + 750, 750)

    for relay_offset in RO:
        gamma, rho, margin = arrange_relay(relay_offset, r_queen, d_queen, r_beam, worker_offset, 1)
        #margin here is a user input: how much overlap of angles do you want when mitigating blindspots?
        #not to be confused with beta margin, wich is the overlap between beta and gamma, and will show up as 'angle' in the code below

        #delta is the angle difference between beta and alpha, d_delta looks for the maximum delta for a range of REFLECTOR offsets
        d_delta = r_queen / (x ** 2 * (r_queen ** 2 / x ** 2 + 1)) - (r_queen + relay_offset) / (x ** 2 * ((r_queen + relay_offset) ** 2 / x ** 2) + 1)
        soff_max = np.where(abs(d_delta) < 0.001)[0]
        soff_max = soff_max[len(soff_max) // 2] # max sting offset, a larger offset would only decrease delta and increase structural mass
        max_delta = abs(np.arctan((r_queen + relay_offset) / soff_max) - np.arctan(r_queen / soff_max))
        alpha_min = abs(np.arctan((r_queen + relay_offset)/soff_max))
        beta_min = alpha_min - max_delta

        d_margin = gamma - beta_min

        ANGLE = np.linspace(margin, margin + d_margin, 10) #range of angle margins for beta
        REFLECTOR_MASS = []# this list will collect the reflector configuration masses for a specific dish depth
        for angle in ANGLE:
            o_s, o_r, r_s, r_r, A_s, A_r, b, a, d = arrange_sting(relay_offset, r_queen, r_beam, 1, gamma, rho, angle)

            if r_r < r_queen:
                extra = r_queen - np.sqrt(r_queen**2 - r_r**2)
                o_r = o_r + extra
                #print(r_r, extra)
            # reflectors' (sting+relay) masses based on, area (with mirrors, radii and circumference (with trusses, for support), and offsets (with trusses)
            reflector_mass = 2 * m_t * (o_s + o_r) + 2 * m_t * (r_r + r_s) + 2 * np.pi * (r_s + r_r) + m_m * (A_r + A_s)
            REFLECTOR_MASS.append(reflector_mass)

        a_i = REFLECTOR_MASS.index(min(REFLECTOR_MASS)) # find optimal margin angle based on minimum reflectors' masses
        angle = ANGLE[a_i]
        MASS.append(min(REFLECTOR_MASS))
        MARGINS.append(angle)

    w_i = MASS.index(min(MASS))

    total_weight = min(MASS) + 0 * m_t * (Queen.length + Worker.length) + 4 * m_t * struct1.length\
                   + (m_f + m_g) * Queen.A + (m_m + m_g) * Worker.A # total mass of the struct for this specific queen depth

    TOTAL_MASS.append(total_weight)
    RELAY_OFFSETS.append(RO[w_i])
    BEST_MARGIN.append(MARGINS[w_i])

i = TOTAL_MASS.index(min(TOTAL_MASS)) # for all evaluated depths and relay offsets, find the optimal configuration

d_best = DEPTHS[i] #depth corresponding to this optimum
offset_best = RELAY_OFFSETS[i] #offset " " " "
margin_best = BEST_MARGIN[i] #margin " " " "

#------------------------------
# configure the final system
d_worker = d_best * r_beam / r_queen

Queen = parabola(r_queen, d_best) # create the big paraboloid
Worker = parabola(r_beam, d_worker, up=0) # create the small paraboloid
worker_offset = Queen.FP - Queen.d + Worker.FP - Worker.d # rim to rim distance beween queen and worker
struct1 = truss(-r_queen, -r_beam, 0, worker_offset)  # 1 truss connecting queen and worker
struct2 = truss(r_beam, r_queen, worker_offset, 0) # 1 other truss connecting queen and worker, (the reason that i create two is mostly just for plotting)

gamma, rho, margin = arrange_relay(offset_best, r_queen, d_best, r_beam, worker_offset, 1)
o_s, o_r, r_s, r_r, A_s, A_r, beta, alpha, delta = arrange_sting(offset_best, r_queen, r_beam, 1, gamma, rho, margin_best)
extra = r_queen - np.sqrt(r_queen**2 - r_r**2)
struct3 = truss(-1, 0, -d_best, -o_s)
struct4 = truss(-(r_queen + o_r), -(r_queen-extra), 0, 0)

mass = min(TOTAL_MASS)

print("Intake radius: ", round(r_queen, 2), "m")
print("Aperture radius: ", round(r_beam, 2), "m")
print("PV cell area: ", round(A_pv, 2), "m²")
print("PV cell disk width: ", round(w_pv, 2), "m")
print("Queen area: ", round(Queen.A, 2), "m²")
print("Worker area: ", round(Worker.A, 2), "m²")
print("Queen depth: ", round(d_best, 2), "m")
print("Worker depth: ", round(d_worker,2), "m")
print("Worker offset: ", round(worker_offset, 2), "m")
print("Sting offset: ", round(o_s, 2), "m")
print("Sting radius: ", round(r_s, 2), "m")
print("Relay offset: ", round(o_r, 2), "m")
print("Relay radius: ", round(r_r, 2), "m")
print("Sting area: ", round(A_s, 2), "m²")
print("Relay area: ", round(A_r, 2), "m²")
print("truss length: ", struct1.length,"m")
print(np.degrees(gamma), np.degrees(rho), np.degrees(beta), np.degrees(alpha), np.degrees(delta))
print("Total mass: ", round(mass, 2), "kg")

print("\nParabola shape: y = -{}*(1 - x²/{}²)\nfor x in [{},{}]".format(round(Queen.d,2), round(Queen.r,2), round(-Queen.r,2), round(Queen.r,2)))

plt.plot(Queen.X, Queen.Y, 'b')
plt.plot(Worker.X, Worker.Y + worker_offset, 'b')
#plt.plot(Queen.FP - Queen.d, '.')
plt.plot(struct1.X, struct1.Y, 'r')
plt.plot(struct2.X, struct2.Y, 'r')
plt.plot(struct3.X, struct3.Y, 'r')
plt.plot(struct4.X, struct4.Y, 'r')
plt.plot([-r_s, r_s], 2 * [-o_s], 'b')
plt.plot(2 * [-(r_queen + o_r)], [-r_r, r_r], 'b')

plt.axis([-1000, 600, -600, 600])
plt.grid()

plt.show()
