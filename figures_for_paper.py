from simulation_functions.KMC_functions import PETRAFTSequenceEnsemble
from analysis_functions.sequence_statistics import ChainLengthDispersity, MonomerFrequency
from analysis_functions.kmer_representation import ConstructGraph
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
# seq = PETRAFTSequenceEnsemble(1000)
# pure = seq.run_statistical(np.array([70.,30.]), 0.01, r_matrix, conv)

# np.savetxt("sample_data/seq_reconstruction/PET-RAFT_pure_seqs.csv",pure)
# plt.imshow(pure)
# plt.show()


########################## symmetrically alternating ############################
# exp_data = 1

# fit = PetRAFTKineticFitting(exp_data, 70, 30)
# A_conv, B_conv = fit.predict_conversion(0.05,0.05)
# conv = np.array([A_conv, B_conv])

# r_matrix = np.array([
#     [0.05, 1],
#     [1, 0.05]
# ])
# seq = PETRAFTSequenceEnsemble(1000)
# alt = seq.run_statistical(np.array([70.,30.]), 0.01, r_matrix, conv)
# np.savetxt("sample_data/seq_reconstruction/PET-RAFT_alt_seqs.csv",alt)
# plt.imshow(alt)
# plt.show()

########################## blocky ############################
# exp_data = 1

# fit = PetRAFTKineticFitting(exp_data, 70, 30)
# A_conv, B_conv = fit.predict_conversion(4.,4.)
# conv = np.array([A_conv, B_conv])

# r_matrix = np.array([
#     [4., 1.],
#     [1., 4.]
# ])
# seq = PETRAFTSequenceEnsemble(1000)
# blocky = seq.run_statistical(np.array([70.,30.]), 0.01, r_matrix, conv)


# np.savetxt("sample_data/seq_reconstruction/PET-RAFT_blocky.csv",blocky)
# plt.imshow(blocky)
# plt.show()

########################## statistical ############################
# exp_data = 1

# fit = PetRAFTKineticFitting(exp_data, 70, 30)
# A_conv, B_conv = fit.predict_conversion(1.,1.)
# conv = np.array([A_conv, B_conv])

# r_matrix = np.array([
#     [1., 1.],
#     [1., 1.]
# ])
# seq = PETRAFTSequenceEnsemble(1000)
# stat = seq.run_statistical(np.array([70.,30.]), 0.01, r_matrix, conv)

# np.savetxt("sample_data/seq_reconstruction/PET-RAFT_stat.csv",stat)

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

target_hex = "#491B4F"
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#FFFFFF", target_hex])

data = np.loadtxt('sample_data/seq_reconstruction/thermal/copolymers/stat.csv', delimiter=' ')
e = ConstructGraph(data, 2)
g = e.get_graph_as_heatmap(num_seq = 1000, segment_size=3)
np.savetxt("sample_data/seq_reconstruction/stat_adj_matrix.csv",g)
plt.imshow(g, cmap=custom_cmap, vmin=0, vmax = 0.45)
plt.colorbar()
plt.savefig("stat_adj_matrix.svg", dpi=300)

# fit.reconstruct_kinetics(0.01,1,1,0.01)