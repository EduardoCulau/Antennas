import numpy as np
import scipy.constants as spc
from antennas import SimulateAntenna, PhasedArrayAntenna

# Parallel Computing
n_jobs = 12

# Define parameters for the phased array antenna
N = 2**10   # Number of elements
d = 0.5     # Distance between elements (in wavelengths)
f = 1e9     # Frequency
k = 2 * np.pi * f/spc.c  # Wave number based on frequency

# Define the angles
angles = np.linspace(0, 2 * np.pi, 360)

paa = SimulateAntenna(PhasedArrayAntenna())
gain = paa.gain(N=N, d=d, k=k, angles=angles, n_jobs=n_jobs, verbose=True)

# Create polar plot
paa.plot()
paa.show()