
import numpy as np
from simulation_functions.KMC_functions1_2 import ThermalRAFTSequenceEnsemble

r_matrix = np.array([
    [1., 1.87, 0.59],
    [1., 1., 0.82],
    [2.02, 1.96, 1.]
])
# listedn in DMA, 


feed = np.array([82.3,41.6,43.1])
conv = np.array([0.97,0.99,0.90])

seq = ThermalRAFTSequenceEnsemble(5000)
seqs = seq.run_statistical(feed, 0.05, r_matrix, 100, conv)

np.savetxt("aug_ktr100kuncap0_01_5000.csv",seqs)