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

Mass Estimations:
Reflectors (sting+relay): 1 area + 4 radial trusses + 1 circumferential truss + 2 connecting trusses to main struct
Queen+worker: 1 area + 4 "radial" trusses + 4 connecting trusses
all much like Chris's drawing

Note:
The code is kinda inefficient and takes around 20 secs, so be patient
"""

# Mass estimates [kg] (as in order of magnitude is in the correct range)
m_t = 1500  # mass of truss per m
m_m = 15  # mass of rigid mirror per m²
m_f = 0.15  # mass of reflective foil per m²

r_queen = 500 # radius of big dish (TBD on power requirement as well --> inner region is gonna be solar cells)
r_beam = 10 # aperture radius: same as beam, same as lens (TBD)

print("Busy making", (r_queen//2)*600*1500*10, "different Honey configurations and calculating their mass so gimme a break OK\n")

DEPTHS = np.linspace(1, r_queen / 2 + 1, r_queen//2) # Range of big dish depths

# setup ranges for mass minimisation
TOTAL_MASS = []
RELAY_OFFSETS = [] # Queen rim to relay distance
BEST_MARGIN = []

for d_queen in DEPTHS:
    d_worker = d_queen * r_beam / r_queen #small dish has same shape as big dish but scaled down, so same

    Queen = parabola(r_queen, d_queen)
    Worker = parabola(r_beam, d_worker, up=0)

    worker_offset = Queen.FP - Queen.d + Worker.FP - Worker.d #rim to rim distance between small and big dish
    struct1 = truss(-r_queen, -r_beam, 0, worker_offset)
    struct2 = truss(r_beam, r_queen, worker_offset, 0)

    RO = np.linspace(0, 600, 600) #range of relay offsets
    MASS = [] # setup lists for reflectors' (sting + relay) masses
    MARGINS = [] # setup list for beta-gamma overlap angles (see later)

    x = np.linspace(d_queen, d_queen + 1500, 1500)

    for relay_offset in RO:
        gamma, rho, margin = arrange_relay(relay_offset, r_queen, d_queen, r_beam, worker_offset, 1)
        #margin here is a user input: how much overlap of angles do you want when mitigating blindspots?
        #not to be confused with beta margin, wich is the overlap between beta and gamma, and will show up as 'angle' in the code below

        #delta is the angle difference between beta and alpha, d_delta looks for the maximum delta for a range of REFLECTOR offsets
        d_delta = r_queen / (x ** 2 * (r_queen ** 2 / x ** 2 + 1)) - (r_queen + relay_offset) / (x ** 2 * ((r_queen + relay_offset) ** 2 / x ** 2) + 1)
        soff_max = np.where(abs(d_delta) < 0.001)[0]
        soff_max = soff_max[len(soff_max) // 2] # max sting offset, a larger offset would only increase both delta and structural mass
        max_delta = abs(np.arctan((r_queen + relay_offset) / soff_max) - np.arctan(r_queen / soff_max))
        alpha_min = abs(np.arctan((r_queen + relay_offset)/soff_max))
        beta_min = alpha_min - max_delta

        d_margin = gamma - beta_min

        ANGLE = np.linspace(margin, margin + d_margin, 10) #range of angle margins for beta
        REFLECTOR_MASS = []
        for angle in ANGLE:
            o_s, o_r, r_s, r_r, A_s, A_r = arrange_sting(relay_offset, r_queen, r_beam, 1, gamma, rho, angle)

            # reflectors' (sting+relay) masses based on, area (with mirrors, radii and circumference (with trusses, for support), and offsets (with trusses)
            reflector_mass = 2 * m_t * (o_s + o_r) + 4 * m_t * (r_r + r_s) + 2 * np.pi * (r_s + r_r) + m_m * (A_r + A_s)
            REFLECTOR_MASS.append(reflector_mass)

        a_i = REFLECTOR_MASS.index(min(REFLECTOR_MASS)) # find optimal margin angle based on minimum reflectors' masses
        angle = ANGLE[a_i]
        MASS.append(min(REFLECTOR_MASS))
        MARGINS.append(angle)

    w_i = MASS.index(min(MASS))

    total_weight = min(MASS) + 4 * m_t * (Queen.length + Worker.length) + 2 * m_t * (struct1.length + struct2.length) \
                   + m_f * (Queen.A + Worker.A) # total mass of the struct for this specific queen depth

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

Queen = parabola(r_queen, d_best)
Worker = parabola(r_beam, d_worker, up=0)
worker_offset = Queen.FP - Queen.d + Worker.FP - Worker.d
struct1 = truss(-r_queen, -r_beam, 0, worker_offset)
struct2 = truss(r_beam, r_queen, worker_offset, 0)

gamma, rho, margin = arrange_relay(offset_best, r_queen, d_best, r_beam, worker_offset, 1)
o_s, o_r, r_s, r_r, A_s, A_r = arrange_sting(offset_best, r_queen, r_beam, 1, gamma, rho, margin_best)

struct3 = truss(-1, 0, -d_best, -o_s)
struct4 = truss(-(r_queen + o_r), -r_queen, 0, 0)

mass = min(TOTAL_MASS)

print("Queen depth: ", d_best // 1, "m")
print("Worker offset: ", worker_offset // 1, "m")
print("Sting offset: ", o_s // 1, "m")
print("Sting radius: ", r_s // 1, "m")
print("Relay offset: ", o_r // 1, "m")
print("Relay radius: ", r_r // 1, "m")
print("Total mass: ", mass // 1, "kg")

print("\nParabola shape: y = {}*(1 - x²/{}²)\nfor x in [{},{}]".format(Queen.d//1, Queen.r//1, -Queen.r//1, Queen.r//1))

plt.plot(Queen.X, Queen.Y, 'b')
plt.plot(Worker.X, Worker.Y + worker_offset, 'b')
plt.plot(Queen.FP - Queen.d, '.')
plt.plot(struct1.X, struct1.Y, 'r')
plt.plot(struct2.X, struct2.Y, 'r')
plt.plot(struct3.X, struct3.Y, 'r')
plt.plot(struct4.X, struct4.Y, 'r')
plt.plot([-r_s, r_s], 2 * [-o_s], 'b')
plt.plot(2 * [-(r_queen + o_r)], [-r_r, r_r], 'b')

plt.axis([-1000, 600, -600, 600])
plt.grid()

plt.show()
