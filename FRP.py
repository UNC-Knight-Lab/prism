
import numpy as np
from simulation_functions.KMC_functions import PETRAFTSequenceEnsemble, ThermalRAFTSequenceEnsemble, FRP

r_matrix = np.array([
    [1., 1.87, 0.59],
    [1., 1., 0.82],
    [2.02, 1.96, 1.]
])
# listedn in DMA, 


feed = np.array([82.3,41.6,43.1])
conv = np.array([0.97,0.99,0.90])

seq = FRP(5000)
seqs = seq.run_statistical(feed, r_matrix, conv)

np.savetxt("FRP_5000.csv",seqs)