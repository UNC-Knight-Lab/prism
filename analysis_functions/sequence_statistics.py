from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx   

class MonomerFrequency():

    def __init__(self, seqs, num_monomers):
        self.sequences = seqs
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

        # Position the nodes using a layout
        pos = nx.spring_layout(G)

        # Calculate node size based on the sum of edge weights (degree)
        node_size = [sum(adj_matrix[i]) * 5 for i in range(len(adj_matrix))]  # Scaling factor of 10 for visibility

        # Calculate edge thickness based on the edge weight
        edges = G.edges(data=True)
        edge_thickness = [d['weight'] / 100 for (u, v, d) in edges]  # Scaling down for visibility

        # Draw the graph with node labels
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=node_size, width=edge_thickness, font_weight='bold')

        # Display the graph
        plt.show()