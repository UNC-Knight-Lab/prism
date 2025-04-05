import numpy as np
import pandas as pd
from analysis_functions.sequence_statistics import ChainLengthDispersity
from simulation_functions.KMC_functions import PETRAFTSequenceEnsemble
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# r_matrix = np.array([
#     [1., 1., 1.96],
#     [1.87, 1., 2.02],
#     [0.59, 0.82, 1.]
# ])

# feed = np.array([82.3,41.6,43.1])
# conv = np.array([0.97,0.99,0.90])

# seq = PETRAFTSequenceEnsemble(500)
# seqs = seq.run_statistical(feed, 0.01, r_matrix,conv)

# np.savetxt("kcap1.csv",seqs)


# d = ChainLengthDispersity(seqs,3)
# masses = np.array([99.13,128.17,169.22])
# c = d.get_dispersity([99.13,128.17,169.22],345.63)

# c = d.get_distribution(masses,345.63)

# plt.figure(figsize=(3,2))
# plt.hist(c,bins=20,weights=np.ones_like(c) / len(c))
# plt.xticks(ticks=range(16000, 27000, 2000))
# plt.savefig("kcap1.pdf",dpi=300)



