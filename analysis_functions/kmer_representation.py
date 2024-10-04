import numpy as np
from collections import defaultdict
import networkx as nx
from matplotlib import pyplot as plt
import itertools

class KmerSimilarity():
    def __init__(self, num_monomers, k):

        self.num_monomers = num_monomers
        self.k = k

    def _kmer_generation(self):
        residues = np.arange(1,self.num_monomers + 1)
        kmers = list(itertools.product(residues, repeat=self.k))
        kmer_mat = np.array(kmers)

        return kmer_mat, len(kmer_mat[:,0])
    
    def _kmer_counting(self, seq, kmer_mat, num_kmers):
        kmer_freq = np.zeros((num_kmers))
        tot_motifs = len(seq) - self.k + 1

        for j in range(tot_motifs):
            test_motif = seq[j:(j+self.k)]

            for k in range(num_kmers):
                if (test_motif == kmer_mat[k,:]).all():
                    kmer_freq[k] += 1
                    break

        return kmer_freq
    
    def _ensemble_kmers(self, seq, kmer_mat, num_kmers):
        all_kmer_freq = np.zeros((seq.shape[0], num_kmers))

        for i in range(seq.shape[0]):
            k_list = self._kmer_counting(seq[i,:], kmer_mat, num_kmers)
            all_kmer_freq[i,:] = k_list / np.sum(k_list)

        return np.sum(all_kmer_freq, axis = 0) / seq.shape[0]
    
    def _KLD(self, s1, s2):

        kl = 0

        for i in range(s1.shape[0]):
            kl += s1[i] * np.log2(s1[i] / s2[i])

        return kl

    def _JSD(self, s1, s2):
        m = (s1 + s2) * 0.5

        kl1 = 0
        for i in range(s1.shape[0]):
            kl1 += s1[i] * np.log2(s1[i] / m[i])
        
        kl2 = 0
        for i in range(s2.shape[0]):
            kl2 += s2[i] * np.log2(s2[i] / m[i])

        return 0.5*kl1 + 0.5*kl2
    
    def frequency(self, seqs1, seqs2):
        kmer_mat, num_kmers = self._kmer_generation()
        s1_freq = self._ensemble_kmers(seqs1, kmer_mat, num_kmers)
        s2_freq = self._ensemble_kmers(seqs2, kmer_mat, num_kmers)

        jsd = self._JSD(s1_freq, s2_freq)
        print(jsd)


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

        plt.show()