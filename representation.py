import numpy as np
import pandas as pd
from analysis_functions.sequence_statistics import MonomerFrequency, EnsembleSimilarity
from simulation_functions.KMC_functions import PETRAFTSequenceEnsemble
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

np.seterr(all='raise')
######################### UNIVERSAL R MATRIX ############################

r_matrix = np.array([
    [1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.]
])

########################## d1 ############################
# d1_feed = np.array([[25., 20., 5.],
#                 [50., 0., 0.]])

# seq = PETRAFTSequenceEnsemble(100)
# d1 = seq.run_block_copolymer(d1_feed, 0.05, r_matrix)

# np.savetxt("d1_seqs.csv",d1)

# ########################## d2 ############################

# d2_feed = np.array([[45., 0., 5.],
#                 [30., 20., 0.]])

# seq = SequenceEnsemble(1000)
# d2 = seq.run_block_copolymer(d2_feed, [0.05, 0.02], r_matrix)
# np.savetxt("d2_seqs.csv",d2)

# ########################## t1 ############################

t1_feed = np.array([[30., 0., 0.],
                [15., 20., 5.],
                [30., 0., 0.]])

seq = PETRAFTSequenceEnsemble(100)
t1 = seq.run_block_copolymer(t1_feed, 0.01, r_matrix)
plt.imshow(t1)
plt.show()
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

# data = np.loadtxt('t4_seqs.csv', delimiter=' ')

# colors = ["#FFFFFF", 
#     "#407ABD",  # Example color for DMA
#     "#491B4F",  # Example color for structural
#     "#EBB101",   # Example color for catalytic
#     "#CB2A57"   # Example color for cap
# ]

# cmap = ListedColormap(colors)
# # plt.figure(figsize=(2.0,0.82))
# plt.imshow(data[-50:,:], cmap=cmap)
# plt.xticks(ticks=range(0, 130, 20))
# # plt.show()
# plt.savefig("t4.svg", format='svg')


########################## chemical patterning plots ############################

# d1 = np.loadtxt('d1_seqs.csv', delimiter=' ')
# d2 = np.loadtxt('d2_seqs.csv', delimiter=' ')
# t1 = np.loadtxt('t1_seqs.csv', delimiter=' ')
# t2 = np.loadtxt('t2_seqs.csv', delimiter=' ')
# t3 = np.loadtxt('t3_seqs.csv', delimiter=' ')
# t4 = np.loadtxt('t4_seqs.csv', delimiter=' ')

# m = MonomerFrequency(t3, 3)
# pattern1 = m.chemical_patterning(features = [-0.4, 4.5, -4.5], method = 'mean')

# m = MonomerFrequency(t4, 3)
# pattern2 = m.chemical_patterning(features = [-0.4, 4.5, -4.5], method = 'mean')

# m = MonomerFrequency(d2, 3)
# pattern2 = m.chemical_patterning(features = [-0.4, 4.5, -4.5], method = 'mean')

# m = MonomerFrequency(d2, 3)
# pattern2 = m.chemical_patterning(features = [-0.4, 4.5, -4.5], method = 'mean')

# plt.figure(figsize=(1.54,1.10))
# plt.plot(pattern1, lw=1)
# plt.plot(pattern2, lw=1)
# plt.yticks(np.arange(-0.5, 3.0, step=0.5))
# plt.xticks(np.arange(0, 140, step=25))
# # plt.show()
# plt.savefig('t3_t4_chemical_patterning.pdf')


########################## sequence heatmap ###########################


# # print(d1)
# all_data = [d1, d2, t1, t2, t3, t4]

# similarity1 = np.zeros((6,6))
# similarity2 = np.zeros((6,6))
# similarity3 = np.zeros((6,6))
# similarity = np.zeros((6,6))


# for s1 in range(6):
#     for s2 in range(s1+1):
#         e = EnsembleSimilarity(all_data[s1], all_data[s2], num_monomers=3)
#         scores = e.global_difference(k=5)
#         # similarity1[s1,s2], similarity2[s1,s2], similarity3[s1,s2] = scores
#         similarity[s1,s2] = scores

# s1 = np.where(similarity1 == 0, np.nan, similarity1)
# s2 = np.where(similarity2 == 0, np.nan, similarity2)
# s3 = np.where(similarity3 == 0, np.nan, similarity3)
# s = np.where(similarity == 0, np.nan, similarity)

# plt.figure()
# plt.imshow(s, cmap='magma_r',vmin=0, vmax=1.0)
# plt.colorbar()
# plt.savefig("fivemonomer.svg", dpi=300)


# plt.figure()
# plt.imshow(s1, cmap='magma_r',vmin=0, vmax=1.0)
# plt.colorbar()
# plt.savefig("onemonomer_1.svg", dpi=300)

# plt.figure()
# plt.imshow(s2, cmap='magma_r',vmin=0, vmax=1.0)
# plt.colorbar()
# plt.savefig("onemonomer_2.svg", dpi=300)

# plt.figure()
# plt.imshow(s3, cmap='magma_r',vmin=0, vmax=1.0)
# plt.colorbar()
# plt.savefig("onemonomer_3.svg", dpi=300)

########################## autocorrelation ###########################
# e = EnsembleSimilarity(d1, d2, num_monomers=3)
# d1, d2 = e.correlation([2,3])

# e = EnsembleSimilarity(t1, t2, num_monomers=3)
# a1, a2 = e.correlation(1)


# e = EnsembleSimilarity(t3, t4, num_monomers=3)
# a3, a4 = e.correlation(1)

#### PAIRWISE
# e = EnsembleSimilarity(d1, d2, num_monomers=3)
# d1, d2 = e.correlation([2,3], corr_type='pair')

# e = EnsembleSimilarity(t1, t2, num_monomers=3)
# a1, a2 = e.correlation([2,3], corr_type='pair')


# e = EnsembleSimilarity(t3, t4, num_monomers=3)
# a3, a4 = e.correlation([2,3], corr_type='pair')

# plt.figure(figsize=(1.54,1.10))
# # plt.plot(a1, lw=1)
# # plt.plot(a2, lw=1)
# plt.plot(a3, lw=1)
# plt.plot(a4, lw=1)
# plt.plot(d1, lw=1)
# plt.plot(d2, lw=1)
# plt.yticks(np.arange(0, 0.25, step=0.05))
# plt.xticks(np.arange(0, 140, step=25))
# # plt.show()
# plt.savefig('pair_corr.pdf', dpi=300)