import numpy as np
import pandas as pd
from analysis_functions.sequence_statistics import MonomerFrequency
from simulation_functions.KMC_functions import SequenceEnsemble
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

######################### UNIVERSAL R MATRIX ############################

r_matrix = np.array([
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]
])

########################## d1 ############################
# d1_feed = np.array([[25., 20., 5.],
#                 [50., 0., 0.]])

# seq = SequenceEnsemble(1000)
# d1 = seq.run_block_copolymer(d1_feed, [0.05, 0.02], r_matrix)
# np.savetxt("d1_seqs.csv",d1)

# ########################## d2 ############################

# d2_feed = np.array([[45., 0., 5.],
#                 [30., 20., 0.]])

# seq = SequenceEnsemble(1000)
# d2 = seq.run_block_copolymer(d2_feed, [0.05, 0.02], r_matrix)
# np.savetxt("d2_seqs.csv",d2)

# ########################## t1 ############################

# t1_feed = np.array([[30., 0., 0.],
#                 [15., 20., 5.],
#                 [30., 0., 0.]])

# seq = SequenceEnsemble(1000)
# t1 = seq.run_block_copolymer(t1_feed, [0.05, 0.02, 0.02], r_matrix)
# np.savetxt("t1_seqs.csv",t1)

# ########################## t2 ############################

# t2_feed = np.array([[17., 10., 2.5],
#                 [41., 0., 0.],
#                 [17., 10., 2.5]])

# seq = SequenceEnsemble(1000)
# t2 = seq.run_block_copolymer(t2_feed, [0.05, 0.02, 0.02], r_matrix)
# np.savetxt("t2_seqs.csv",t2)

# ########################## t3 ############################

# t3_feed = np.array([[25., 0., 5.],
#                 [20., 20., 0.],
#                 [30., 0., 0.]])

# seq = SequenceEnsemble(1000)
# t3 = seq.run_block_copolymer(t3_feed, [0.05, 0.02, 0.02], r_matrix)
# np.savetxt("t3_seqs.csv",t3)

# ########################## t4 ############################

# t4_feed = np.array([[10., 20., 0.],
#                 [40., 0., 0.],
#                 [25., 0., 5.]])

# seq = SequenceEnsemble(1000)
# t4 = seq.run_block_copolymer(t4_feed, [0.05, 0.02, 0.02], r_matrix)
# np.savetxt("t4_seqs.csv",t4)




########################## HEATMAP GENERATION ############################

data = np.loadtxt('t4_seqs.csv', delimiter=' ')

colors = ["#FFFFFF", 
    "#407ABD",  # Example color for DMA
    "#491B4F",  # Example color for structural
    "#EBB101",   # Example color for catalytic
    "#CB2A57"   # Example color for cap
]

cmap = ListedColormap(colors)
# plt.figure(figsize=(2.0,0.82))
plt.imshow(data[-50:,:], cmap=cmap)
plt.xticks(ticks=range(0, 130, 20))
# plt.show()
plt.savefig("t4.svg", format='svg')