from simulation_functions.KMC_functions import SequenceEnsemble
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
from analysis_functions.sequence_statistics import ChainLengthDispersity, MonomerFrequency
from analysis_functions.kmer_representation import ConstructGraph
import pandas as pd

r_matrix = np.array([
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]
])

# all feed ratio columns given as DMA, OA and tBA/AA
g1_feed = np.array([
    [30.5, 0.6, 0.8],
    [14.2, 1.5, 1.6],
    [7.8, 1.7, 1.8],
    [6.6, 2.1, 2.9],
    [3.9, 2.4, 4.7],
    [2.9, 1.9, 3.9],
    [3.2, 2.2, 4.7],
    [5.9, 3.6, 9.5]
])

g2_feed = np.array([
    [17.8, 0.5, 0.8],
    [11.3, 0.7, 0.9],
    [6.8, 1.0, 0.8],
    [7.3, 1.5, 2.4],
    [9.8, 4.2, 6.8],
    [7.0, 2.7, 5.5],
    [19.9, 7.6, 18.5],
])

g3_feed = np.array([
    [25.5, 0.5, 0.8],
    [14.8, 1.1, 1.9],
    [4.6, 0.7, 1.0],
    [3.8, 0.8, 1.5],
    [6.9, 3.4, 6.2],
    [6.6, 3.4, 6.4],
    [4.2, 2.1, 4.3],
    [8.7, 4.2, 9.7]
])

diblock_feed = np.array([
    [77.7, 0.0, 0.0],
    [0.52, 16.4, 30.8]
])

stat_feed = np.array([74.3, 17.3, 34.6])

# seq = SequenceEnsemble(100)
# g3 = seq.run_gradient_copolymer(g3_feed, 0.05, r_matrix)

# seq = SequenceEnsemble(100)
# diblock = seq.run_block_copolymer(diblock_feed, [0.05,0.05], r_matrix)

# seq = SequenceEnsemble(100)
# stat = seq.run_statistical(stat_feed, 0.05, r_matrix)

colors = ["#FFFFFF", 
    "#B1CAE4",  # Example color for 1
    "#491B4F",  # Example color for 2
    "#24957B",  # Example color for 3
    "#CB2A57"   # Example color for 4
]


# cmap = ListedColormap(colors)
# plt.figure(figsize=(2.0,0.82))
# plt.imshow(g3[50:,:], cmap=cmap)
# plt.xticks(ticks=range(0, 181, 20))
# plt.savefig("matt_paper/speckle_plots/g3.svg", format='svg', dpi=300)

# m = MonomerFrequency(stat, 3)
# m.frequency_output()

data = pd.read_csv('matt_paper/g3_freq.csv')
plt.plot(data.iloc[:,1])
plt.plot(data.iloc[:,2])
plt.plot(data.iloc[:,3])
plt.ylim([0,1.0])
plt.xlim([0,180])
plt.savefig('matt_paper/freq_plots/g3.pdf', dpi=300)

# e = ConstructGraph(g1, 3)
# e.get_graph_as_heatmap(num_seq = 1000, segment_size=2)