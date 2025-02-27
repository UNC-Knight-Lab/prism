from distutils.ccompiler import new_compiler
from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import mode

class MonomerFrequency():

    def __init__(self, seqs, num_monomers):
        self.sequences = seqs.astype(int)
        self.num_seqs = seqs.shape[0]
        self.max_DP = seqs.shape[1]
        self.num_monomers = num_monomers
    
    def _mode_chemical_features(self, features):
        replace_dict = {}

        replace_dict[0] = 0

        for i in range(self.num_monomers):
            replace_dict[i + 1] = features[i]

        modes, _ = mode(self.sequences, axis=0)

        replace_func = np.vectorize(lambda x: replace_dict.get(x, x))

        return replace_func(modes[0])

    def _mean_chemical_features(self, features):
        replace_dict = {}

        replace_dict[0] = 0

        for i in range(self.num_monomers):
            replace_dict[i+1] = features[i]

        self.sequences = np.vectorize(lambda x: replace_dict.get(x, x))(self.sequences)
        
        return np.mean(self.sequences, axis = 0)
    
    def chemical_patterning(self, features, method = 'mode'):
        if method == 'mean':
            pattern = self._mean_chemical_features(features)
        else:
            pattern = self._mode_chemical_features(features)
        
        return pattern
    
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
        plt.ylim([0,1.0])
        plt.savefig('freq.svg',format='svg')
    
    def frequency_output(self):
        freq_count = self.get_frequency()

        df = pd.DataFrame(freq_count).T
        df.to_csv('freq.csv')

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
    
    def _list_of_kmers(self, k):
        l = [dist for dist in product(range(k + 1), repeat=self.num_monomers) if sum(dist) == k]
        return np.array(l)

    def _coarse_graining(self, seq, k, kmer_list):

        num_beads = np.int64(seq.shape[1] / k)

        cg_seq = np.zeros((num_beads, len(kmer_list)))

        total_junctions = 0

        for i in range(seq.shape[0]):
            running_idx = 0

            # print(seq[i,:])

            for j in range(0,seq.shape[1], k):

                segment = seq[i,j:(j+k)]
                seg_counts = np.bincount(segment, minlength=self.num_monomers + 1)
                seg_counts = seg_counts[1:]

                # print(segment, seg_counts)

                for idx, dist in enumerate(kmer_list):
                    if np.array_equal(dist, seg_counts):
                        cg_seq[running_idx, idx] += 1
                        running_idx += 1
                        total_junctions += 1
                        break
                else:
                    break
                
        return cg_seq / total_junctions
    
    def _cosine_sim(self, s1, s2):
        return np.dot(s1, s2) / (np.linalg.norm(s1) * np.linalg.norm(s2))
    
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
    
    def global_difference(self, k = 1):

        if k == 1:
            m1 = self._count_monomers(self.seqs1)
            m2 = self._count_monomers(self.seqs2)

            m1, m2 = self._trim(m1, m2)
            m1, m2 = self._realign(m1,m2)

            m2_flip = np.flip(m2)

            for i in range(self.num_monomers):
                p = max(self._cosine_sim(m1[i,:], m2[i,:]), self._cosine_sim(m1[i,:], m2_flip[i,:]))
                print(p)
        else:
            kmer_comp = self._list_of_kmers(k)
            print(kmer_comp)

            cg_s1 = self._coarse_graining(self.seqs1, k, kmer_comp)
            cg_s2 = self._coarse_graining(self.seqs2, k, kmer_comp)

            for i in range(kmer_comp.shape[0]):
                for j in range(i+1):
                    f = self._cosine_sim(cg_s1[:,i], cg_s2[:,j]) * self._cosine_sim(kmer_comp[i,:], kmer_comp[j,:])
                    print(i, j, f)