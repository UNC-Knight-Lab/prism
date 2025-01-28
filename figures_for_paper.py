from simulation_functions.KMC_functions import SequenceEnsemble
from analysis_functions.sequence_statistics import ChainLengthDispersity, MonomerFrequency
from analysis_functions.kmer_representation import ConstructGraph
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from fitting_functions.ODE_solving import PetRAFTKineticFitting

########################## pure gradient ##########################
# exp_data = 1

# fit = PetRAFTKineticFitting(exp_data, 70, 30)
# A_conv, B_conv = fit.predict_conversion(5.,0.2)
# conv = np.array([A_conv, B_conv])

# r_matrix = np.array([
#     [5., 1.],
#     [1, 0.2]
# ])
# seq = SequenceEnsemble(1000)
# pure = seq.run_statistical(np.array([70.,30.]), 0.05, r_matrix, conv)

# np.savetxt("pure_seqs.csv",pure)

########################## symmetrically alternating ############################
# exp_data = 1

# fit = PetRAFTKineticFitting(exp_data, 70, 30)
# A_conv, B_conv = fit.predict_conversion(0.01,0.01)
# conv = np.array([A_conv, B_conv])

# r_matrix = np.array([
#     [0.01, 1],
#     [1, 0.01]
# ])
# seq = SequenceEnsemble(1000)
# alt = seq.run_statistical(np.array([70.,30.]), 0.05, r_matrix, conv)

# np.savetxt("alt_seqs.csv",alt)

########################## blocky ############################
# exp_data = 1

# fit = PetRAFTKineticFitting(exp_data, 70, 30)
# A_conv, B_conv = fit.predict_conversion(4.,4.)
# conv = np.array([A_conv, B_conv])

# r_matrix = np.array([
#     [4., 1.],
#     [1., 4.]
# ])
# seq = SequenceEnsemble(1000)
# blocky = seq.run_statistical(np.array([70.,30.]), 0.05, r_matrix, conv)

# np.savetxt("blocky.csv",blocky)

########################## statistical ############################
# exp_data = 1

# fit = PetRAFTKineticFitting(exp_data, 70, 30)
# A_conv, B_conv = fit.predict_conversion(1.,1.)
# conv = np.array([A_conv, B_conv])

# r_matrix = np.array([
#     [1., 1.],
#     [1., 1.]
# ])
# seq = SequenceEnsemble(1000)
# stat = seq.run_statistical(np.array([70.,30.]), 0.05, r_matrix, conv)

# np.savetxt("stat.csv",stat)

####################################################################


# colors = ["#FFFFFF", 
#     "#407ABD",  # Example color for 1
#     "#491B4F",  # Example color for 2
#     "#CB2A57"   # Example color for 3
# ]

# cmap = ListedColormap(colors)
# # plt.figure(figsize=(2.0,0.82))
# plt.imshow(alt[50:,:], cmap=cmap)
# plt.xticks(ticks=range(0, 141, 20))
# plt.savefig("alt.svg", format='svg')

# m = MonomerFrequency(stat, 2)
# m.plot_frequency()

data = np.loadtxt('stat.csv', delimiter=' ')
e = ConstructGraph(data, 2)
e.get_graph_as_heatmap(num_seq = 1000, segment_size=3)

# fit.reconstruct_kinetics(0.01,1,1,0.01)