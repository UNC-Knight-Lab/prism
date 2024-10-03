from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx   
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.stats import pearsonr

class MonomerFrequency():

    def __init__(self, seqs, num_monomers, idx):
        self.sequences = seqs
        self.num_seqs = seqs.shape[0]
        self.max_DP = seqs.shape[1]
        self.num_monomers = num_monomers
        self.idx_f = idx
    
    def get_frequency(self):
        freq_count = np.zeros((self.num_monomers, self.max_DP))

        for i in range(1, self.num_monomers+1):
            freq_count[i-1,:] = np.sum(self.sequences == i, axis = 0) / self.num_seqs
        
        return freq_count

    def plot_frequency(self):
        freq_count = self.get_frequency()

        for i in range(self.num_monomers):
            plt.plot(freq_count[i,:])
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

class ConstructGraph():

    def __init__(self, seqs, num_monomers):
        self.sequences = seqs
        self.num_seqs = seqs.shape[0]
        self.max_DP = seqs.shape[1]
        self.num_monomers = num_monomers
    
    def _nearest_neighbors(self):

        connect_mat = np.zeros((self.num_monomers, self.num_monomers))

        for i in range(self.num_seqs):
            for j in range(self.max_DP - 1):
                n = int(self.sequences[i, j])
                nn = int(self.sequences[i, j+1])

                if n == 4 or nn == 4:
                    break
                else:
                    connect_mat[n-1, nn-1] += 1
        
        return connect_mat

    def get_graph(self):
        adj_matrix = self._nearest_neighbors()

        # Create a weighted graph from the adjacency matrix
        G = nx.from_numpy_array(adj_matrix)

        pos = nx.spring_layout(G)

        node_size = [sum(adj_matrix[i]) * 5 for i in range(len(adj_matrix))]  # Scaling factor of 10 for visibility

        edges = G.edges(data=True)
        edge_thickness = [d['weight'] / 100 for (u, v, d) in edges]  # Scaling down for visibility

        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=node_size, width=edge_thickness, font_weight='bold')

        # Display the graph
        plt.show()

class KmerSimilarity():
    def __init__(self, seqs1, seqs2, num_monomers, idx):

        self.seqs1 = seqs1 
        self.seqs2 = seqs2
        self.num_monomers = num_monomers
        self.idx = idx

    def _count_kmers(self, sequence, k):

        kmer_counts = defaultdict(int)
        
        for i in range(len(sequence) - k + 1):
            kmer = tuple(sequence[i:i + k])
            kmer_counts[kmer] += 1
        
        return kmer_counts

    def _get_unified_kmers(self, *unique_kmers_list):
        unified_kmers = set()
        
        for unique_kmers in unique_kmers_list:
            for kmer in unique_kmers:
                unified_kmers.add(tuple(kmer))
        
        return sorted(unified_kmers)

    def _vectorize_kmers(self, unique_kmers, unified_kmers):
        frequency_vector = np.array([unique_kmers.get(kmer, 0) for kmer in unified_kmers])
        return frequency_vector

    def _normalize_vector(self, vector):
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def _euclidean_distance(self, vec1, vec2):
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)

        return euclidean_distances(vec1, vec2)[0][0]

    def _pearson_correlation(self, vec1, vec2):

        return pearsonr(vec1, vec2)[0]
    
    def _similarity(self, seq1, seq2, k):
        kmers1  = self._count_kmers(seq1, k)
        kmers2 = self._count_kmers(seq2, k)
        
        unified_kmers = self._get_unified_kmers(kmers1, kmers2)
        
        vector1 = self._vectorize_kmers(kmers1, unified_kmers)
        vector2 = self._vectorize_kmers(kmers2, unified_kmers)
        
        norm_vec1 = self._normalize_vector(vector1)
        norm_vec2 = self._normalize_vector(vector2)

        a = self._euclidean_distance(norm_vec1, norm_vec2)
        # b = self._pearson_correlation(norm_vec1, norm_vec2)

        return a
    
    def sequence_alignment(self):

        res = np.zeros((100))

        for i in range(100):
            res[i] = self._similarity(self.seqs1[i,:], self.seqs2[i,:], 5)
        
        print(res)

class GlobalSimilarity():

    def __init__(self, seqs1, seqs2, num_monomers, idx):
        self.seqs1 = seqs1 
        self.seqs2 = seqs2
        self.num_monomers = num_monomers
        self.idx = idx
    
    def _global_alignment(self, seq1, seq2):
        match_score = 1
        mismatch_score = -1
        gap_penalty = -2

        n = len(seq1)
        m = len(seq2)
        
        # Create the score matrix
        score_matrix = np.zeros((n + 1, m + 1))
        
        # Initialize the first row and column
        for i in range(1, n + 1):
            score_matrix[i, 0] = score_matrix[i - 1, 0] + gap_penalty
        for j in range(1, m + 1):
            score_matrix[0, j] = score_matrix[0, j - 1] + gap_penalty
        
        # Fill the score matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    score_matrix[i][j] = score_matrix[i - 1, j - 1] + match_score
                else:
                    score_matrix[i, j] = max(
                        score_matrix[i - 1, j] + gap_penalty,     # Gap in seq2
                        score_matrix[i, j - 1] + gap_penalty,     # Gap in seq1
                        score_matrix[i - 1, j - 1] + mismatch_score # Mismatch
                    )
        
        return score_matrix[-1, -1]  # Return the final alignment score
    
    def sequence_alignment(self):

        res = np.zeros((100))

        for i in range(100):
            res[i] = self._global_alignment(self.seqs1[i,:], self.seqs2[i,:])
        
        print(res)


# class EnsembleSimilarity():

#     def __init__(self, seqs1, seqs2, num_monomers, idx):

#         if isinstance(seqs1, pd.DataFrame) == False:
#             self.seqs1 = pd.DataFrame(seqs1)
#         else:
#             self.seqs1 = seqs1
        
#         if isinstance(seqs2, pd.DataFrame) == False:
#             self.seqs2 = pd.DataFrame(seqs2)
#         else:
#             self.seqs2 = seqs2

#         self.num_monomers = num_monomers
#         self.idx = idx
    
#     def _replace(self):

#         dict_replace = {}

#         for i in range(1, self.num_monomers+1):
#             if i in self.idx:
#                 dict_replace[i] = 1
#             else:
#                 dict_replace[i] = 0
        
#         self.seqs1.replace(dict_replace, inplace=True)
#         self.seqs2.replace(dict_replace, inplace=True)

#         self.seqs1 = self.seqs1.loc[:, (self.seqs1 != 0).any(axis=0)]
#         self.seqs2 = self.seqs2.loc[:, (self.seqs2 != 0).any(axis=0)]

    
#     def _trim(self):
#         if self.seqs1.shape[1] > self.seqs2.shape[1]:
#             final_index = self.seqs2.shape[1]

#             diff = int((self.seqs1.shape[1] - final_index))

#             s1_p = self.seqs1.mean(axis = 0).iloc[:-diff]
#             s2_p = self.seqs2.mean(axis = 0)

#         else: # s1 is smaller than s2
#             final_index = self.seqs1.shape[1]

#             diff = int((self.seqs2.shape[1] - final_index))

#             s1_p = self.seqs1.mean(axis = 0)
#             s2_p = self.seqs2.mean(axis = 0).iloc[:-diff]
        
#         return s1_p, s2_p
    
#     def _KLD(self, s1, s2):

#         kl = 0

#         for i in range(s1.shape[0]):
#             kl += s1.iloc[i] * np.log2(s1.iloc[i] / s2.iloc[i])

#         return kl
    
#     def KL_divergence(self): # "foreground" monomer
#         self._replace()
#         s1, s2 = self._trim()
#         kl = self._KLD(s1, s2)

#         return kl

#     def JS_divergence(self):
#         self._replace()

#         s1, s2 = self._trim()

#         s1 /= sum(s1)
#         s2 /= sum(s2)

#         m = (s1 + s2.values) * 0.5

#         return 0.5*self._KLD(s1, m) + 0.5*self._KLD(s1, m)