import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import itertools
import pandas as pd

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

        plt.plot(s1_freq)
        plt.plot(s2_freq)
        plt.show()

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
        connections = 0

        for i in range(self.num_seqs):
            for j in range(self.max_DP - 1):
                n = int(self.sequences[i, j])
                nn = int(self.sequences[i, j+1])

                if n == (self.num_monomers + 1) or nn == (self.num_monomers + 1):
                    break
                else:
                    connect_mat[n-1, nn-1] += 1
                    connections += 1
        
        return connect_mat / connections, np.arange(1, self.num_monomers + 1)
    
    def _kmer_generation(self, k):
        residues = np.arange(1,self.num_monomers + 1)
        kmers = list(itertools.product(residues, repeat=k))
        kmer_mat = np.array(kmers)

        return kmer_mat, len(kmer_mat[:,0])
    
    def _kmer_counting(self, seq, kmer_mat, num_kmers, k):
        kmer_freq = np.zeros((num_kmers, num_kmers))
        connections = 0

        for i in range(0,num_kmers, k): # not sliding window for connectivity
            test_motif = seq[i:(i+k)]
            nn_motif = seq[(i+k):(i + (2*k))]

            checkpoint1 = False
            checkpoint2 = False

            for j in range(num_kmers):
                if (test_motif == kmer_mat[j,:]).all():
                    test_index = j
                    checkpoint1 = True
                
                if (nn_motif == kmer_mat[j,:]).all():
                    nn_index = j
                    checkpoint2 = True
                
                if checkpoint1 == True and checkpoint2 == True:
                    kmer_freq[test_index, nn_index] += 1
                    connections += 1
                    break
                    
        return kmer_freq / connections
    
    def _convert_kmer_labels(self, kmer_mat):
        labels = {}

        if kmer_mat.ndim == 1:
            for i, row in enumerate(kmer_mat):
                labels[i] =  str(kmer_mat[i])
        else:
            for i, row in enumerate(kmer_mat):
                labels[i] =  ''.join(map(str, kmer_mat[i,:]))
        
        return labels

    def _convert_to_graph(self, adj_matrix, labels):
        G = nx.DiGraph()

        # G = nx.from_numpy_array(adj_matrix)
        n = adj_matrix.shape[0]

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    G.add_edge(i, i, weight=adj_matrix[i,j])
                else:
                    G.add_edge(i, j,weight=adj_matrix[i,j])
                    G.add_edge(j, i, weight=adj_matrix[j,i])
        
        # # Create positions in a regular polygon
        pos = {}
        radius = 0.01  # You can adjust this for larger/smaller polygons
        for i in range(n):
            angle = 2 * np.pi * i / n
            pos[i] = (radius * np.cos(angle), radius * np.sin(angle))

        # pos = nx.spring_layout(G, seed = 5)
        fig, ax = plt.subplots()
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color='black',ax=ax)
        node_size = [sum(adj_matrix[i]) * 1000 for i in range(len(adj_matrix))]  # Scaling factor of 10 for visibility
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=node_size, ax=ax)

        # curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
        # straight_edges = list(set(G.edges()) - set(curved_edges))
        # nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
        edges = G.edges(data=True)
        edge_thickness = [d['weight'] * 8 for (u, v, d) in edges] 
        arc_rad = 0.25
        nx.draw_networkx_edges(G, pos, ax=ax, connectionstyle=f'arc3, rad = {arc_rad}', width=edge_thickness)

        # self._draw_edges(G, pos, adj_matrix, node_size)
        # self._draw_middle_arrows(G, pos, adj_matrix)
        # plt.show()
        plt.savefig('graph.svg',dpi=300, bbox_inches='tight', transparent=True)
    
    def _monomer_graph(self):
        adj_matrix, kmer_vec = self._nearest_neighbors()
        labels = self._convert_kmer_labels(kmer_vec)
        print(adj_matrix)
        self._convert_to_graph(adj_matrix, labels)

    def get_graph(self, num_seq, segment_size = 1):

        if segment_size == 1:
            self._monomer_graph()
        else:
            kmer_mat, num_kmers = self._kmer_generation(segment_size)
            labels = self._convert_kmer_labels(kmer_mat)
            adj_matrix = np.zeros((num_kmers, num_kmers))

            for i in range(self.sequences.shape[0]):
                t = self._kmer_counting(self.sequences[i,:],kmer_mat, num_kmers, segment_size)
                adj_matrix += (t / num_seq)
            
            self._convert_to_graph(adj_matrix, labels)

            np.savetxt("adjacency_matrix.csv",adj_matrix)
    
    def get_graph_as_heatmap(self, num_seq, segment_size = 1):
        if segment_size == 1:
            self._monomer_graph()
        else:
            kmer_mat, num_kmers = self._kmer_generation(segment_size)
            labels = self._convert_kmer_labels(kmer_mat)
            adj_matrix = np.zeros((num_kmers, num_kmers))

            for i in range(self.sequences.shape[0]):
                t = self._kmer_counting(self.sequences[i,:],kmer_mat, num_kmers, segment_size)
                adj_matrix += (t / num_seq)
        
        return adj_matrix
