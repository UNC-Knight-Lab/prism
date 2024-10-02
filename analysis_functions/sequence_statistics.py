from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx   
import pandas as pd

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

class EnsembleSimilarity():

    def __init__(self, seqs1, seqs2, num_monomers, idx):

        if isinstance(seqs1, pd.DataFrame) == False:
            self.seqs1 = pd.DataFrame(seqs1)
        else:
            self.seqs1 = seqs1
        
        if isinstance(seqs2, pd.DataFrame) == False:
            self.seqs2 = pd.DataFrame(seqs2)
        else:
            self.seqs2 = seqs2

        self.num_monomers = num_monomers
        self.idx = idx
    
    def _replace(self):

        dict_replace = {}

        for i in range(1, self.num_monomers+1):
            if i in self.idx:
                dict_replace[i] = 1
            else:
                dict_replace[i] = 0
        
        self.seqs1.replace(dict_replace, inplace=True)
        self.seqs2.replace(dict_replace, inplace=True)

        self.seqs1 = self.seqs1.loc[:, (self.seqs1 != 0).any(axis=0)]
        self.seqs2 = self.seqs2.loc[:, (self.seqs2 != 0).any(axis=0)]

    
    def _trim(self):
        if self.seqs1.shape[1] > self.seqs2.shape[1]:
            final_index = self.seqs2.shape[1]

            diff = int((self.seqs1.shape[1] - final_index))

            s1_p = self.seqs1.mean(axis = 0).iloc[:-diff]
            s2_p = self.seqs2.mean(axis = 0)

        else: # s1 is smaller than s2
            final_index = self.seqs1.shape[1]

            diff = int((self.seqs2.shape[1] - final_index))

            s1_p = self.seqs1.mean(axis = 0)
            s2_p = self.seqs2.mean(axis = 0).iloc[:-diff]
        
        return s1_p, s2_p
    
    def _KLD(self, s1, s2):

        kl = 0

        for i in range(s1.shape[0]):
            kl += s1.iloc[i] * np.log2(s1.iloc[i] / s2.iloc[i])

        return kl
    
    def KL_divergence(self): # "foreground" monomer
        self._replace()
        s1, s2 = self._trim()
        kl = self._KLD(s1, s2)

        return kl

    def JS_divergence(self):
        self._replace()

        s1, s2 = self._trim()

        s1 /= sum(s1)
        s2 /= sum(s2)

        m = (s1 + s2.values) * 0.5

        return 0.5*self._KLD(s1, m) + 0.5*self._KLD(s1, m)