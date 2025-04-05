import pandas as pd
from fitting_functions.ODE_solving import PetRAFTKineticFitting
import numpy as np
from matplotlib import pyplot as plt
from analysis_functions.sequence_statistics import ChainLengthDispersity
from simulation_functions.KMC_functions import PETRAFTSequenceEnsemble, FRP
from fitting_functions.traditional_methods import MeyerLoweryFitting
import seaborn as sns

####### SAMPLE RATES PLOTS #########

# exp_data = pd.read_excel('sample_data/two_monomer_systems/pdb-5-070.xlsx')
# p = PetRAFTKineticFitting(exp_data, 70., 30.)

# # p.reconstruct_kinetics(0.01,0.01,20)
# # print(p.predict_conversion(0.001,0.001,20))
# Aconv, Bconv, totconv = p.test_values(0.2,5,30)

# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 8

# plt.figure(figsize=(3,2))
# plt.plot(totconv, Aconv)
# plt.plot(totconv, Bconv)
# plt.xlim([0, 1])
# plt.ylabel('monomer conversion')
# plt.xlabel('total conversion')
# plt.tight_layout()
# # plt.show()
# plt.savefig('0_2__5.pdf',dpi=300)




####

# r_matrix = np.array([
#     [1., 1., 1.96],
#     [1.87, 1., 2.02],
#     [0.59, 0.82, 1.]
# ])

# feed = np.array([82.3,41.6,43.1])
# conv = np.array([0.97,0.99,0.90])

# seq = PETRAFTSequenceEnsemble(100)
# seqs = seq.run_statistical(feed, 0.01, r_matrix,conv)

# np.savetxt("Ctr_100_aug.csv",seqs)


# df1 = np.loadtxt('sample_data/molecular_weight/ktr100kuncap0_01_5000.csv', delimiter=' ')
# df2 = np.loadtxt('sample_data/molecular_weight/ktr100kuncap0_01_5000_2.csv', delimiter=' ')
# seqs = np.concatenate((df1, df2), axis=0)
# d = ChainLengthDispersity(seqs,3)
# masses = np.array([99.13,128.17,169.22])
# c = d.get_dispersity([99.13,128.17,169.22],345.63)
# print(c)
# c1 = d.get_distribution(masses,345.63)

# df1 = np.loadtxt('sample_data/molecular_weight/FRP_5000.csv', delimiter=' ')
# df2 = np.loadtxt('sample_data/molecular_weight/FRP_5000_2.csv', delimiter=' ')
# seqs = np.concatenate((df1, df2), axis=0)
# d = ChainLengthDispersity(seqs,3)
# masses = np.array([99.13,128.17,169.22])
# c = d.get_dispersity([99.13,128.17,169.22],345.63)
# print(c)
# c2 = d.get_distribution(masses,345.63)


seqs = np.loadtxt('sample_data/molecular_weight/ktr100kuncap0_01_5000.csv', delimiter=' ')
d = ChainLengthDispersity(seqs,3)
masses = np.array([99.13,128.17,169.22])
c = d.get_dispersity([99.13,128.17,169.22],345.63)
print(c)
c1 = d.get_distribution(masses,345.63)

seqs = np.loadtxt('sample_data/molecular_weight/aug_ktr100kuncap0_01_5000.csv', delimiter=' ')
d = ChainLengthDispersity(seqs,3)
masses = np.array([99.13,128.17,169.22])
c = d.get_dispersity([99.13,128.17,169.22],345.63)
print(c)
c2 = d.get_distribution(masses,345.63)


# df = pd.DataFrame([c1,c2,c3],columns=['k0','k500','k100'])

# plt.figure(figsize=(3,2))

# sns.histplot(
#     c1, element="step",
#     stat="probability",alpha=0.05
# )

sns.histplot(
    c1, element="step",
    alpha=0.05
)

sns.histplot(
    c2, element="step",
    alpha=0.05
)

plt.show()
# plt.savefig("full_vs_augmented.pdf",dpi=300)

# c = d.get_dispersity([99.13,128.17,169.22],345.63)

####

# exp_data = pd.read_excel('sample_data/two_monomer_systems/PDB-5-076_ML.xlsx')
# p = MeyerLoweryFitting()
# # p.extract_rates(exp_data, 1.62, 0.44)
# fA, y, conv = p.visualize_overlay(exp_data, 1.84, 0.26)

# plt.figure(figsize=(3,2))
# plt.plot(fA, y,lw=1)
# plt.scatter(fA, conv,s=10)
# plt.xlabel("fA")
# plt.ylabel('total conversion')
# plt.xlim([0,1])
# plt.tight_layout()
# # plt.show()
# plt.savefig("MeyerLowery76.pdf", dpi=300)