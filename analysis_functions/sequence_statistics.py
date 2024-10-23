from distutils.ccompiler import new_compiler
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr

class MonomerFrequency():

    def __init__(self, seqs, num_monomers):
        self.sequences = seqs.astype(int)
        self.num_seqs = seqs.shape[0]
        self.max_DP = seqs.shape[1]
        self.num_monomers = num_monomers
    
    def get_frequency(self):
        freq_count = np.zeros((self.num_monomers, self.max_DP))

        for i in range(1, self.num_monomers+1):
            freq_count[i-1,:] = np.sum(self.sequences == i, axis = 0) / self.num_seqs
        
        return freq_count

    def plot_frequency(self):
        freq_count = self.get_frequency()

        for i in range(self.num_monomers):
            plt.plot(freq_count[i,:])

        plt.xlabel('degree of polymerization')
        plt.ylabel('monomer probability')
        plt.xlim(left=2)
        plt.show()

class ChainLengthDispersity():

    def __init__(self, seqs, num_monomers):
        self.sequences = seqs
        self.num_seqs = seqs.shape[0]
        self.max_DP = seqs.shape[1]
        self.num_monomers = num_monomers
    
    def _calc_weight(self, vec, mass):
        mon_idx = np.arange(1, self.num_monomers+1)

        freq = np.zeros((self.num_monomers))

        for i in range(self.num_monomers):
            freq[i] = np.sum(vec == mon_idx[i])
        
        return np.dot(freq, mass)
    
    def get_dispersity(self, mass):
        chain_mass = np.zeros((self.num_seqs))

        for i in range(self.num_seqs):
            chain_mass[i] = self._calc_weight(self.sequences[i,:], mass)
        
        unique_elements, counts = np.unique(chain_mass, return_counts=True)

        counts = counts / np.sum(counts)

        M_n = np.dot(unique_elements, counts)
        w_n = (unique_elements * counts) / M_n
        M_w = np.dot(unique_elements, w_n)

        return M_w / M_n
    
    def get_distribution(self, mass):
        chain_mass = np.zeros((self.num_seqs))

        for i in range(self.num_seqs):
            chain_mass[i] = self._calc_weight(self.sequences[i,:], mass)
        
        plt.hist(chain_mass, bins=20)
        plt.show()

class EnsembleSimilarity():

    def __init__(self, seqs1, seqs2, num_monomers):
        self.seqs1 = seqs1.astype(int)
        self.seqs2 = seqs2.astype(int)
        self.num_monomers = num_monomers
    
    def _trim(self, mat1, mat2):

        non_zero_columns = ~np.all(mat1 == 0, axis=0)
        mat1 = mat1[:, non_zero_columns]

        non_zero_columns = ~np.all(mat2 == 0, axis=0)
        mat2 = mat2[:, non_zero_columns]

        return mat1, mat2

    def _coarse_graining(self, seq, k):
        idx = 0
        num_beads = len(seq) - k + 1

        cg_seq = np.zeros((num_beads, self.num_monomers))

        for i in range(0,num_beads):
            segment = seq[i:(i+k)]

            for j in range(1,self.num_monomers+1):
                cg_seq[i,j-1] = np.count_nonzero(segment == j)

        return cg_seq, num_beads
    
    def _cosine_sim(self, s1, s2):
        return np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
    
    def _global_alignment(self, seq1, seq2):
        threshold = 0.2
        cg_s1, n_beads1 = self._coarse_graining(seq1, 5)
        cg_s2, n_beads2 = self._coarse_graining(seq2, 5)

        n_beads = min(n_beads1, n_beads2)

        scores_mat = np.zeros((n_beads))

        for i in range(n_beads):
            if (np.all(cg_s1[i,:] == 0) == False) and (np.all(cg_s2[i,:] == 0) == False):
                scores_mat[i] = 1 - self._cosine_sim(cg_s1[i,:], cg_s2[i,:])

        return scores_mat

    def _count_monomers(self, seq):

        maxDP = seq.shape[1]
        num_seqs = seq.shape[0]
        m = np.zeros((self.num_monomers, maxDP))
        
        for i in range(num_seqs):
            for j in range(maxDP):
                monomer = seq[i,j]
                
                if monomer <= self.num_monomers and monomer > 0:
                    m[monomer - 1, j] += 1

        m /= num_seqs

        return m
    
    def _realign(self, s1, s2):
        if s1.shape[1] > s2.shape[1]:
            final_index = s2.shape[1]
            diff = int((s1.shape[1] - final_index))

            s1 = s1[:,:-diff]

        else: # s1 is smaller than s2
            final_index = s1.shape[1]

            diff = int((s2.shape[1] - final_index))
            s2 = s2[:,:-diff]
        
        return s1, s2
    
    def global_difference(self):
        m1 = self._count_monomers(self.seqs1)
        m2 = self._count_monomers(self.seqs2)

        m1, m2 = self._trim(m1, m2)
        m1, m2 = self._realign(m1,m2)

        for i in range(self.num_monomers):
            p = self._cosine_sim(m1[i,:], m2[i,:])
            print(p)
    
    def sequence_alignment(self):

        seq1, seq2 = self._trim(self.seqs1, self.seqs2)
        s = []

        for i in range(100):
            p = self._global_alignment(seq1[i,:], seq2[i,:])
            s.append(p)

        scores_mat = np.vstack(s)
        scores = np.sum(scores_mat, axis=0) / 100
        print(np.sum(scores))
        plt.plot(scores)
        plt.show()