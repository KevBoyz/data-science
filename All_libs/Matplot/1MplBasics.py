from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np

x = np.linspace(0, 8*np.pi, 100)
fig, axs3 = plt.subplot_mosaic([['upleft'],
                                ['lowright']], layout='constrained')

sin = np.sin(x)
cos = np.cos(x)
tan = np.tan(x)
cosh = np.cosh(x)

axs4 = axs3['upleft'].secondary_xaxis('top', functions=(np.rad2deg, np.deg2rad))
axs4 = axs3['lowright'].secondary_xaxis('top', functions=(np.rad2deg, np.deg2rad))


axs3['upleft'].plot(sin)
axs3['lowright'].plot(cos)


axs3.show()

