from simulation_functions.KMC_functions import SequenceEnsemble
from fitting_functions.ODE_solving import PetRAFTKineticFitting
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

seq = SequenceEnsemble(100)
feed_ratios = np.array([
    [17.6, 0., 0.],
    [11.3, 0.48, 0.85],
    [6.8, 1.15, 1.77],
    [7.3, 2.11, 2.58],
    [9.8, 4.16, 6.78],
    [7.0, 2.71, 5.53],
    [19.9, 7.58, 18.48],
])

r_matrix = np.array([
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]
])


all_seqs = seq.run_gradient_copolymer(feed_ratios, 0.05, r_matrix)
cmap = plt.get_cmap('viridis',7)
bounds = np.linspace(0,5,6)
norm = mcolors.BoundaryNorm(bounds, cmap.N)
plt.imshow(all_seqs, cmap=cmap, norm=norm)
plt.colorbar()
plt.show()

# exp_data = pd.read_csv('/Users/suprajachittari/Documents/peter/sequence/PDB-4-98_fractions.csv', header=0)

# f = PetRAFTKineticFitting(exp_data, 97.95, 34.29) # for Peter
# # f = PetRAFTKineticFitting(exp_data, 116.88, 21.83) # for Bridgette
# f.extract_rates(1, 2, 1, 10)
# # f.display_overlay([1, 2, 1, 10])
