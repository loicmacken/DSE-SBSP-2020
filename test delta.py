import numpy as np
import matplotlib.pyplot as plt
from numpy.core._multiarray_umath import ndarray

x = np.linspace(1,2000,2000)

R = 500
d = 150

delta = np.arctan((R + d) / x) - np.arctan(R / x)
delta2 = np.arctan(((R + d) / x - R / x) / (1 + (R + d) * R / x**2))

d_delta = (1 + d**2 / x**2)**(-1) - (1 + R**2 / x**2)**(-1)
d_delta2 = (1 + (((R + d) / x - R / x) / (1 + (R + d) * R / x**2))**2)**(-1)
d_delta3 = R / (x**2 * (R**2/x**2 + 1)) - (R + d)/(x**2 * ((R + d)**2 / x**2) + 1)

best = np.where(abs(d_delta3) < 0.0001)[0]
best = best[len(best)//2]



#plt.plot(x, delta2)
plt.plot(x, delta)

plt.plot(x, d_delta2,'r')
plt.plot(x, d_delta3,'b')
plt.show()