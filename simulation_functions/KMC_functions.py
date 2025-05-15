from re import L
from unicodedata import digit
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

volume = 1
frac_capped = 0.95

class PETRAFTSequenceEnsemble():

    def __init__(self, n_chains):
        self.n_chains = n_chains

    def _get_rate_constants(self, C_tr, c_uncap, rate_matrix):
        self.k_tr = np.mean(rate_matrix)
        self.k_tr *= C_tr
        self.k_uncap = c_uncap * np.mean(rate_matrix)
        self.k_terminate = 0.1
        self.k_init = 1000
    
    def _fragmentation(self, CTA_mmol, Z_mmol, R_mmol, delta):
        CTA_mmol -= delta
        Z_mmol += delta
        R_mmol += delta

        return CTA_mmol, R_mmol, Z_mmol

    def _first_monomer(self, mmol_feed, monomer_index):
        u = np.random.random()*np.sum(mmol_feed)

        if u < mmol_feed[0]:
            return monomer_index[0]
        else:
            for i in range(1,self.num_monomers):
                if u > np.sum(mmol_feed[:i]) and u <= np.sum(mmol_feed[:i+1]):
                    return monomer_index[i]
    
    def _growth_update(self, move, mmol_feed, delta):

        last_monomer = ((move // 10) % 10) # true index
        adding_monomer = (int(abs(move * 10)) % 10)

        indices = np.argwhere(self.chain_status == last_monomer)
        chain = int(random.choice(indices))

        self.sequences[chain, self.lengths[chain]] = adding_monomer
        mmol_feed[int(adding_monomer - 1)] -= delta
        self.lengths[chain] += 1
        self.chain_status[chain] = adding_monomer
        
        return mmol_feed
    
    def _first_growth(self, mmol_feed, new, idx, delta, monomer_index):
        for i in range(self.num_monomers):
            if new == monomer_index[i] and ((mmol_feed[i] - delta) >= 0):
                self.sequences[idx, self.lengths[idx]] = monomer_index[i]
                mmol_feed[i] -= delta
                self.lengths[idx] += 1
                self.chain_status[idx] = monomer_index[i]
                break
        
        return mmol_feed
    
    def _initiate_with_R(self, mmol_feed, monomer_index, R_mmol, idx, delta):
        new = self._first_monomer(mmol_feed, monomer_index)

        for i in range(self.num_monomers):
            if new == monomer_index[i] and ((mmol_feed[i] - delta) >= 0):
                self.sequences[idx, self.lengths[idx]] = monomer_index[i]
                mmol_feed[i] -= delta
                R_mmol -= delta
                self.lengths[idx] += 1
                self.chain_status[idx] = monomer_index[i]
                break
        
        return mmol_feed, R_mmol

    def _draw_uninitated_chain(self):
        indices = np.argwhere(self.chain_status == 0)

        if indices.shape[0] == 0:
            return False
        else:
            return random.choice(indices)
    
    def _get_last_monomer(self, chain):
        return self.sequences[chain, int(self.lengths[chain] - 1)]
    
    def _draw_uncapped_chain(self):
        indices = np.where((self.chain_status >= 1) & (self.chain_status <= self.num_monomers))[0]

        return random.choice(indices)

    def _draw_capped_chain(self, capped_index):
        indices = np.argwhere(self.chain_status == capped_index)

        return random.choice(indices)
    
    def _capping_move(self, capped_chains, Z_mmol, delta):
        cap_chain = self.k_tr*capped_chains*delta
        cap_Z = self.k_tr*Z_mmol

        u = np.random.random()*(cap_chain + cap_Z)

        if u < cap_chain:
            return 'capped by chain'
        else:
            return 'capped by Z group'

    def _uncapping_move(self, uncapped_chains, pc, delta):
        cap_chain = self.k_uncap*uncapped_chains*delta
        cap_pc = self.k_uncap*pc

        u = np.random.random()*(cap_chain + cap_pc)

        if u < cap_chain:
            return 'uncapped by chain'
        else:
            return 'uncapped by photocatalyst'   


    def _capping_update(self, chain, capping_index):
        self.sequences[chain, self.lengths[chain]] = capping_index
        self.lengths[chain] += 1
        self.chain_status[chain] = capping_index

    
    def _uncapping_update(self, chain, uncapped_index):    
        self.sequences[chain, self.lengths[chain] - 1] = uncapped_index
        self.lengths[chain] -= 1
        self.chain_status[chain] = self._get_last_monomer(chain)
    
    def _Z_group_capping(self, Z_mmol, delta):
        if Z_mmol - delta > 0:
            Z_mmol -= delta
        
        return Z_mmol
    
    def _photocatalyst_uncapping(self, idx, Z_mmol, delta, uncapped_index):

        self.sequences[idx, self.lengths[idx]] = uncapped_index
        Z_mmol += delta
        self.lengths[idx] -= 1
        self.chain_status[idx] = self._get_last_monomer(idx)
        
        return Z_mmol
    
    def _chain_termination(self, chain, R_mmol, delta, dead_index):
        if R_mmol - delta > 0:

            self.sequences[chain, self.lengths[chain]] = dead_index
            R_mmol -= delta
            self.lengths[chain] += 1
            self.chain_status[chain] = dead_index
        
        return R_mmol
                
    def _add_chains(self, init, capped_chains):
        new_chains = int(init * capped_chains)
        new = np.zeros((new_chains, self.max_DP))

        self.sequences = np.vstack((new, self.sequences))
        self.lengths = np.concatenate([np.zeros((new_chains), dtype=int), self.lengths])
        self.n_chains = self.sequences.shape[0]
        self.chain_status = np.concatenate([np.zeros((new_chains)), self.chain_status])
    
    def _terminate_uncapped(self, dead_index):
        uncapped = np.argwhere(self.chain_status == 1)

        for idx in uncapped:
            self.sequences[idx, self.lengths[idx]] = dead_index
            self.chain_status[idx] = 3
    
    def _num_chains(self, indices):
        return len(indices) if indices.size > 0 else 0

    def _system_moves(self, pc, CTA_mmol, R_mmol, Z_mmol, delta, r_ij, mmol_feed, monomer_index, capped_chains, uncapped_chains):

        rates = np.zeros(((self.num_monomers**2)+5,3)) #change

        idx = 0

        # free chains undergo growth
        for i in range(self.num_monomers):
            i_chains = np.argwhere(self.chain_status == (i+1))
            terminated_in_i = self._num_chains(i_chains)
            for j in range(self.num_monomers):
                rates[idx,0] = r_ij[i,j] * mmol_feed[j] * (terminated_in_i * delta)
                rates[idx,1] = ((monomer_index[i])*10) + ((monomer_index[j]) * 0.1) # last residue above 10, adding residue below 10
                rates[idx,2] = sum(rates[:,0])

                idx += 1

        # free chains get capped by Z group or another capped chain
        rates[idx,0] = self.k_tr * ((Z_mmol * (uncapped_chains)) + ((capped_chains*delta) * (uncapped_chains*delta)))
        rates[idx,1] = self.num_monomers + 1
        rates[idx,2] = sum(rates[:,0])

        idx += 1

        # free chain terminates
        rates[idx,0] = self.k_terminate * R_mmol * (uncapped_chains*delta)
        rates[idx,1] = self.num_monomers + 2
        rates[idx,2] = sum(rates[:,0])

        idx += 1

        # capped chain uncaps by another capped chain or photocatalyst
        rates[idx,0] = self.k_uncap * ((pc * (capped_chains * delta)) + ((capped_chains*delta) * (uncapped_chains*delta)))
        rates[idx,1] = self.num_monomers + 3
        rates[idx,2] = sum(rates[:,0])

        idx += 1

        # R group initiates chain
        rates[idx,0] = self.k_init * sum(mmol_feed) * R_mmol
        rates[idx,1] = self.num_monomers + 4
        rates[idx,2] = sum(rates[:,0])

        idx += 1

        # photochatalyst splits CTA
        rates[idx,0] = self.k_init * CTA_mmol * pc
        rates[idx,1] = self.num_monomers + 5
        rates[idx,2] = sum(rates[:,0])

        u = np.random.random()*rates[-1,2]
        # print(rates)
        if u <= rates[0, 2]:
            return rates[0,1]
        else:
            for i in range(1,self.num_monomers**2 + 5):
                if rates[i-1,2] < u <= rates[i,2]:
                    return rates[i,1]


    def _run_first_block(self, mmol_feed, pc, r_matrix, conversion): # first through self.num_monomers are monomers, next is capped, next is dead
        delta = 1 / self.n_chains
        CTA_mmol = 1.
        R_mmol = 0. 
        Z_mmol = 0.
        orig = mmol_feed.copy()

        uncapped_index = 0
        monomer_indexes = np.arange(1, self.num_monomers+1)
        capped_index = self.num_monomers + 1
        dead_index = capped_index + 1

        left_over = (1 - conversion) * mmol_feed
    
        attempt = 1

        while np.max(self.lengths) <= self.max_DP:
            uncapped = np.where((self.chain_status >= 1) & (self.chain_status <= self.num_monomers))[0]
            uncapped_chains = self._num_chains(uncapped)
            
            capped = np.where(self.chain_status == capped_index)[0]
            capped_chains = self._num_chains(capped)
           
            move = self._system_moves(pc, CTA_mmol, R_mmol, Z_mmol, delta, r_matrix, mmol_feed, monomer_indexes, capped_chains, uncapped_chains)

            if move == self.num_monomers + 1: # chain caps any mechanism
                cap_move = self._capping_move(capped_chains, Z_mmol, delta)

                chain = self._draw_uncapped_chain()
                self._capping_update(chain, capped_index)

                if cap_move == 'capped by chain':
                    swap_chain = self._draw_capped_chain(capped_index)
                    self._uncapping_update(swap_chain, uncapped_index)
                else:
                    Z_mmol = self._Z_group_capping(Z_mmol, delta)

            elif move == self.num_monomers + 2: # chain terminates
                chain = self._draw_uncapped_chain()
                R_mmol = self._chain_termination(chain, R_mmol, delta, dead_index)

            elif move == self.num_monomers + 3: # chain uncaps
                uncap_move = self._uncapping_move(uncapped_chains, pc, delta)

                if uncap_move == 'uncapped by chain':
                    swap_chain = self._draw_capped_chain(capped_index)
                    self._uncapping_update(swap_chain, uncapped_index)

                    chain = self._draw_uncapped_chain()
                    self._capping_update(chain, capped_index)
                else:
                    chain = self._draw_capped_chain(capped_index)
                    Z_mmol = self._photocatalyst_uncapping(chain, Z_mmol, delta, uncapped_index)

            elif move == self.num_monomers + 4: # new chain initiates                                      
                chain = self._draw_uninitated_chain()
                mmol_feed, R_mmol = self._initiate_with_R(mmol_feed, monomer_indexes, R_mmol, chain, delta)
            
            elif move == self.num_monomers + 5: # CTA breaks apart
                CTA_mmol, R_mmol, Z_mmol = self._fragmentation(CTA_mmol, Z_mmol, R_mmol, delta)

            else: # growth move
                mmol_feed = self._growth_update(move, mmol_feed, delta)
            

            if attempt % 10000 == 0:
                print(1 - np.sum(mmol_feed) / np.sum(orig), capped_chains / self.n_chains)

            attempt += 1

            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break
    
    def run_statistical(self, feed_ratios, pc, rate_matrix, c_tr, c_uncap, conversion = None):
        self._get_rate_constants(c_tr, c_uncap, rate_matrix)

        
        self.num_monomers = feed_ratios.shape[0]
        self.max_DP = int(np.sum(feed_ratios) + 1000)
        self.sequences = np.zeros((self.n_chains, self.max_DP),dtype=int)
        self.lengths = np.zeros((self.n_chains), dtype=int)
        self.chain_status = np.zeros((self.n_chains))

        if conversion is None:
            conversion = np.ones(self.num_monomers)

        capped_chains = 0

        self._run_first_block(feed_ratios, pc, rate_matrix, conversion)

        return self.sequences


class ThermalRAFTSequenceEnsemble():

    def __init__(self, n_chains):
        self.n_chains = n_chains

    def _get_rate_constants(self, C_tr, C_uncap, rate_matrix):
        self.k_tr = np.mean(rate_matrix)
        self.k_tr *= C_tr
        self.k_uncap = C_uncap * np.mean(rate_matrix)
        self.k_terminate = 0.1
        self.k_init = 100

    def _first_monomer(self, mmol_feed, monomer_index):
        u = np.random.random()*np.sum(mmol_feed)

        if u < mmol_feed[0]:
            return monomer_index[0]
        else:
            for i in range(1,self.num_monomers):
                if u > np.sum(mmol_feed[:i]) and u <= np.sum(mmol_feed[:i+1]):
                    return monomer_index[i]
    
    def _full_growth_update(self, move, mmol_feed, delta):

        last_monomer = ((move // 10) % 10) # true value
        adding_monomer = (int(abs(move * 10)) % 10)

        indices = np.argwhere(self.chain_status == last_monomer)
        chain = int(random.choice(indices))

        self.sequences[chain, self.lengths[chain]] = adding_monomer
        mmol_feed[int(adding_monomer - 1)] -= delta
        self.lengths[chain] += 1
        self.chain_status[chain] = adding_monomer
        
        return mmol_feed
    
    def _first_growth(self, mmol_feed, new, idx, delta, monomer_index):
        for i in range(self.num_monomers):
            if new == monomer_index[i] and ((mmol_feed[i] - delta) >= 0):
                self.sequences[idx, self.lengths[idx]] = monomer_index[i]
                mmol_feed[i] -= delta
                self.lengths[idx] += 1
                self.chain_status[idx] = monomer_index[i]
                break
        
        return mmol_feed
    
    def _initiate_with_R(self, mmol_feed, monomer_index, R_mmol, idx, delta):
        new = self._first_monomer(mmol_feed, monomer_index)

        for i in range(self.num_monomers):
            if new == monomer_index[i] and ((mmol_feed[i] - delta) >= 0):
                self.sequences[idx, self.lengths[idx]] = monomer_index[i]
                mmol_feed[i] -= delta
                R_mmol -= delta
                self.lengths[idx] += 1
                self.chain_status[idx] = monomer_index[i]
                break
        
        return mmol_feed, R_mmol

    def _draw_uninitated_chain(self):
        indices = np.argwhere(self.chain_status == 0)

        if indices.shape[0] == 0:
            return False
        else:
            return random.choice(indices)
    
    def _get_last_monomer(self, chain):
        return self.sequences[chain, int(self.lengths[chain] - 1)]
    
    def _draw_uncapped_chain(self):
        indices = np.where((self.chain_status >= 1) & (self.chain_status <= self.num_monomers))[0]

        return random.choice(indices)

    def _draw_capped_chain(self, capped_index):
        indices = np.argwhere(self.chain_status == capped_index)

        return random.choice(indices)
    
    def _capping_move(self, capped_chains, CTA_mmol, delta):
        cap_chain = self.k_tr*capped_chains*delta
        cap_CTA = self.k_tr*CTA_mmol

        u = np.random.random()*(cap_chain + cap_CTA)

        if u < cap_chain:
            return 'capped by chain'
        else:
            return 'capped by CTA'


    def _capping_update(self, chain, capping_index):
        self.sequences[chain, self.lengths[chain]] = capping_index
        self.lengths[chain] += 1
        self.chain_status[chain] = capping_index

    
    def _uncapping_update(self, chain, uncapped_index):    
        self.sequences[chain, self.lengths[chain] - 1] = uncapped_index
        self.lengths[chain] -= 1
        self.chain_status[chain] = self._get_last_monomer(chain)

    
    def _chain_termination(self, chain, R_mmol, delta, dead_index):
        if R_mmol - delta > 0:

            self.sequences[chain, self.lengths[chain]] = dead_index
            R_mmol -= delta
            self.lengths[chain] += 1
            self.chain_status[chain] = dead_index
        
        return R_mmol
                
    def _add_chains(self, init, capped_chains):
        new_chains = int(init * capped_chains)
        new = np.zeros((new_chains, self.max_DP))

        self.sequences = np.vstack((new, self.sequences))
        self.lengths = np.concatenate([np.zeros((new_chains), dtype=int), self.lengths])
        self.n_chains = self.sequences.shape[0]
        self.chain_status = np.concatenate([np.zeros((new_chains)), self.chain_status])
    
    def _terminate_uncapped(self, dead_index):
        uncapped = np.argwhere(self.chain_status == 1)

        for idx in uncapped:
            self.sequences[idx, self.lengths[idx]] = dead_index
            self.chain_status[idx] = 10
    
    def _num_chains(self, indices):
        return len(indices) if indices.size > 0 else 0
    
    def _CTA_capping(self, CTA_mmol, R_mmol, delta):
        CTA_mmol -= delta
        R_mmol += delta
        
        return CTA_mmol, R_mmol

    def _system_moves(self, CTA_mmol, R_mmol, delta, r_ij, mmol_feed, monomer_index, capped_chains, uncapped_chains):

        rates = np.zeros(((self.num_monomers**2)+4,3)) #change

        idx = 0

        # free chains undergo growth
        for i in range(self.num_monomers):
            i_chains = np.argwhere(self.chain_status == (i+1))
            terminated_in_i = self._num_chains(i_chains)
            for j in range(self.num_monomers):
                rates[idx,0] = r_ij[i,j] * mmol_feed[j] * (terminated_in_i * delta)
                rates[idx,1] = ((monomer_index[i])*10) + ((monomer_index[j]) * 0.1) # last residue above 10, adding residue below 10
                rates[idx,2] = sum(rates[:,0])

                idx += 1

        # free chains get capped by CTA group or another capped chain
        rates[idx,0] = self.k_tr * ((CTA_mmol * (uncapped_chains)) + ((capped_chains*delta) * (uncapped_chains*delta)))
        rates[idx,1] = self.num_monomers + 1
        rates[idx,2] = sum(rates[:,0])

        idx += 1

        # free chain terminates
        rates[idx,0] = self.k_terminate * R_mmol * (uncapped_chains*delta)
        rates[idx,1] = self.num_monomers + 2
        rates[idx,2] = sum(rates[:,0])

        idx += 1

        # capped chain uncaps by another capped chain
        rates[idx,0] = self.k_uncap * ((capped_chains*delta) * (uncapped_chains*delta))
        rates[idx,1] = self.num_monomers + 3
        rates[idx,2] = sum(rates[:,0])

        idx += 1

        # R group initiates chain
        rates[idx,0] = self.k_init * sum(mmol_feed) * R_mmol
        rates[idx,1] = self.num_monomers + 4
        rates[idx,2] = sum(rates[:,0])

        idx += 1
        # print(rates)
        u = np.random.random()*rates[-1,2]

        if u <= rates[0, 2]:
            return rates[0,1]
        else:
            for i in range(1,self.num_monomers**2 + 4):
                if rates[i-1,2] < u <= rates[i,2]:
                    return rates[i,1]


    def _force_growth(self, chain, mmol_feed, r_matrix, monomer_index):
        last_monomer = self.sequences[chain, (self.lengths[chain] - 1)]

        rates_add = np.zeros((self.num_monomers,3)) # first column is normalized rate, second is addition, third is sum

        for i in range(self.num_monomers):
            if last_monomer == monomer_index[i]:
                for j in range(self.num_monomers):
                    rates_add[j,0] = r_matrix[i,j] * mmol_feed[j]
                    rates_add[j,1] = monomer_index[j]
                    rates_add[j,2] = sum(rates_add[:,0])
        
        # random number generation for move
        u = np.random.random()*rates_add[-1,2]

        if u <= rates_add[0, 2]:
            return rates_add[0,1]
        else:
            for i in range(1,self.num_monomers + 3):
                if rates_add[i-1,2] < u <= rates_add[i,2]:
                    return rates_add[i,1]
    
    def _abridged_growth_update(self, mmol_feed, new, idx, delta, monomer_index):

        for i in range(self.num_monomers):
            if new == monomer_index[i] and ((mmol_feed[i] - delta) >= 0):
                self.sequences[idx, self.lengths[idx]] = monomer_index[i]
                mmol_feed[i] -= delta
                self.lengths[idx] += 1
                self.chain_status[idx] = monomer_index[i]
                break
        
        return mmol_feed
                
    def _run_abridged(self, mmol_feed, initiator, r_matrix, conversion): # first through self.num_monomers are monomers, next is capped, next is dead
        delta = (1 + initiator) / self.n_chains
        CTA_mmol = 1.
        R_mmol = 0. 
        orig = mmol_feed.copy()

        n_impure_chains = int(initiator * self.n_chains)
        uncapped_index = 0
        monomer_indexes = np.arange(1, self.num_monomers+1)
        capped_index = self.num_monomers + 1
        dead_index = capped_index + 1

        left_over = (1 - conversion) * mmol_feed
    
        attempt = 1
        capped_chains = 0
        uncapped_chains = 0

        for i in range(n_impure_chains):
            new = self._first_monomer(mmol_feed, monomer_indexes)
            mmol_feed = self._abridged_growth_update(mmol_feed, new, i, delta, monomer_indexes)
            
        uncapped_chains += n_impure_chains

        while (capped_chains / self.n_chains) < frac_capped:
            uncapped = np.where((self.chain_status >= 1) & (self.chain_status <= self.num_monomers))[0]
            uncapped_chains = self._num_chains(uncapped)
            
            capped = np.where(self.chain_status == capped_index)[0]
            capped_chains = self._num_chains(capped)
           
            move = self._system_moves(CTA_mmol, R_mmol, delta, r_matrix, mmol_feed, monomer_indexes, capped_chains, uncapped_chains)

            if move == self.num_monomers + 1: # chain caps any mechanism
                cap_move = self._capping_move(capped_chains, CTA_mmol, delta)

                if cap_move == 'capped by chain':
                    chain = self._draw_uncapped_chain()
                    self._capping_update(chain, capped_index)
                    swap_chain = self._draw_capped_chain(capped_index)
                    self._uncapping_update(swap_chain, uncapped_index)
                else:
                    if CTA_mmol > delta:
                        chain = self._draw_uncapped_chain()
                        self._capping_update(chain, capped_index)
                        CTA_mmol, R_mmol = self._CTA_capping(CTA_mmol, R_mmol, delta)
                
            elif move == self.num_monomers + 2: # chain terminates
                chain = self._draw_uncapped_chain()
                R_mmol = self._chain_termination(chain, R_mmol, delta, dead_index)

            elif move == self.num_monomers + 3: # chain uncaps
                    swap_chain = self._draw_capped_chain(capped_index)
                    self._uncapping_update(swap_chain, uncapped_index)

                    chain = self._draw_uncapped_chain()
                    self._capping_update(chain, capped_index)

            elif move == self.num_monomers + 4: # new chain initiates                                      
                if R_mmol > delta:
                    if capped_chains + uncapped_chains < self.n_chains:
                        chain = self._draw_uninitated_chain()
                        mmol_feed, R_mmol = self._initiate_with_R(mmol_feed, monomer_indexes, R_mmol, chain, delta)
                    else:
                        R_mmol = 0.
            
            else: # growth move
                mmol_feed = self._full_growth_update(move, mmol_feed, delta)
            
            if attempt % 1000 == 0:
                print(1 - np.sum(mmol_feed) / np.sum(orig), capped_chains / self.n_chains, CTA_mmol)

            attempt += 1

            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break
        
        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == capped_index:
                self._uncapping_update(chain, uncapped_index)
                new = self._force_growth(chain, mmol_feed, r_matrix, monomer_indexes)
                mmol_feed = self._abridged_growth_update(mmol_feed, new, chain, delta, monomer_indexes)

                swap_chain = self._draw_uncapped_chain()
                self._capping_update(swap_chain, capped_index)
            elif self.chain_status[chain] == 0:
                continue
            else:
                new = self._force_growth(chain, mmol_feed, r_matrix, monomer_indexes)
                mmol_feed = self._abridged_growth_update(mmol_feed, new, chain, delta, monomer_indexes)
                self._capping_update(chain, capped_index)
                swap_chain = self._draw_capped_chain(capped_index)
                self._uncapping_update(swap_chain, uncapped_index)
    
            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break

        return capped_chains, dead_index
    
    def _run_first_block(self, mmol_feed, initiator, r_matrix, conversion): # first through self.num_monomers are monomers, next is capped, next is dead
        delta = (1 + initiator) / self.n_chains
        CTA_mmol = 1.
        R_mmol = 0. 
        orig = mmol_feed.copy()
        n_impure_chains = int(initiator * self.n_chains)

        uncapped_index = 0
        monomer_indexes = np.arange(1, self.num_monomers+1)
        capped_index = self.num_monomers + 1
        dead_index = capped_index + 1

        left_over = (1 - conversion) * mmol_feed
    
        attempt = 1
        capped_chains = 0
        uncapped_chains = 0

        for i in range(n_impure_chains):
            new = self._first_monomer(mmol_feed, monomer_indexes)
            mmol_feed = self._abridged_growth_update(mmol_feed, new, i, delta, monomer_indexes)
            
        uncapped_chains += n_impure_chains

        while np.max(self.lengths) <= self.max_DP:
            uncapped = np.where((self.chain_status >= 1) & (self.chain_status <= self.num_monomers))[0]
            uncapped_chains = self._num_chains(uncapped)

            capped = np.where(self.chain_status == capped_index)[0]
            capped_chains = self._num_chains(capped)
           
            move = self._system_moves(CTA_mmol, R_mmol, delta, r_matrix, mmol_feed, monomer_indexes, capped_chains, uncapped_chains)

            if move == self.num_monomers + 1: # chain caps any mechanism
                cap_move = self._capping_move(capped_chains, CTA_mmol, delta)

                if cap_move == 'capped by chain':
                    chain = self._draw_uncapped_chain()
                    self._capping_update(chain, capped_index)
                    swap_chain = self._draw_capped_chain(capped_index)
                    self._uncapping_update(swap_chain, uncapped_index)
                else:
                    if CTA_mmol > delta:
                        chain = self._draw_uncapped_chain()
                        self._capping_update(chain, capped_index)
                        CTA_mmol, R_mmol = self._CTA_capping(CTA_mmol, R_mmol, delta)
                
            elif move == self.num_monomers + 2: # chain terminates
                chain = self._draw_uncapped_chain()
                R_mmol = self._chain_termination(chain, R_mmol, delta, dead_index)

            elif move == self.num_monomers + 3: # chain uncaps
                swap_chain = self._draw_capped_chain(capped_index)
                self._uncapping_update(swap_chain, uncapped_index)

                chain = self._draw_uncapped_chain()
                self._capping_update(chain, capped_index)

            elif move == self.num_monomers + 4: # new chain initiates
                if R_mmol > delta:
                    if capped_chains + uncapped_chains < self.n_chains:
                        chain = self._draw_uninitated_chain()
                        mmol_feed, R_mmol = self._initiate_with_R(mmol_feed, monomer_indexes, R_mmol, chain, delta)
                    else:
                        R_mmol = 0.
            
            else: # growth move
                mmol_feed = self._full_growth_update(move, mmol_feed, delta)
            
            if attempt % 1000 == 0:
                print(1 - np.sum(mmol_feed) / np.sum(orig), capped_chains / self.n_chains)

            attempt += 1

            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break

        return capped_chains, dead_index
    
    def _run_block(self, mmol_feed, inits, rate_matrix, capped_chains, conversion):
        delta = 1.0 / capped_chains
        num_initiated_chains = int(inits * capped_chains)

        uncapped_index = 0
        monomer_indexes = np.arange(1, self.num_monomers+1)
        capped_index = self.num_monomers + 1

        left_over = (1 - conversion) * mmol_feed

        for i in range(num_initiated_chains):
            new = self._first_monomer(mmol_feed, self.num_monomers, monomer_indexes)
            mmol_feed = self._abridged_growth_update(mmol_feed, new, i, delta, monomer_indexes, self.num_monomers)
        
        self.chain_status[0:num_initiated_chains] = 1

        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,(self.n_chains + num_initiated_chains)))

            if self.chain_status[chain] == 1:
                new = self._force_growth(chain, mmol_feed, rate_matrix, self.num_monomers, monomer_indexes)
                mmol_feed = self._abridged_growth_update(mmol_feed, new, chain, delta, monomer_indexes, self.num_monomers)
                self._capping_update(chain, capped_index)
                swap_chain = self._draw_capped_chain()
                self._uncapping_update(swap_chain, uncapped_index)
            else:
                self._uncapping_update(chain, uncapped_index)
                new = self._force_growth(chain, mmol_feed, rate_matrix, self.num_monomers, monomer_indexes)
                mmol_feed = self._abridged_growth_update(mmol_feed, new, chain, delta, monomer_indexes, self.num_monomers)

                swap_chain = self._draw_uncapped_chain()
                self._capping_update(swap_chain, capped_index)
            
            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break
        
        return capped_chains
    
    def _run_gradient(self, mmol_feed, rate_matrix, conversion):
        delta = 1.0 / self.num_monomers

        uncapped_index = 0
        monomer_indexes = np.arange(1, self.num_monomers+1)
        capped_index = self.num_monomers + 1

        left_over = (1 - conversion) * mmol_feed

        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 1:
                new = self._force_growth(chain, mmol_feed, rate_matrix, self.num_monomers, monomer_indexes)
                mmol_feed = self._abridged_growth_update(mmol_feed, new, chain, delta, monomer_indexes, self.num_monomers)
                self._capping_update(chain, capped_index)
                swap_chain = self._draw_capped_chain()
                self._uncapping_update(swap_chain, uncapped_index)
            else:
                self._uncapping_update(chain, uncapped_index)
                new = self._force_growth(chain, mmol_feed, rate_matrix, self.num_monomers, monomer_indexes)
                mmol_feed = self._abridged_growth_update(mmol_feed, new, chain, delta, monomer_indexes, self.num_monomers)

                swap_chain = self._draw_uncapped_chain()
                self._capping_update(swap_chain, capped_index)
            
            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break
    

    def run_statistical(self, feed_ratios, initiator, rate_matrix, c_tr, c_uncap, conversion = None, sim = 'full'):
        self._get_rate_constants(c_tr, c_uncap, rate_matrix)

        self.num_monomers = feed_ratios.shape[0]
        self.max_DP = int(np.sum(feed_ratios) + 1000)
        self.sequences = np.zeros((self.n_chains, self.max_DP),dtype=int)
        self.lengths = np.zeros((self.n_chains), dtype=int)
        self.chain_status = np.zeros((self.n_chains))

        if conversion is None:
            conversion = np.ones(self.num_monomers)

        capped_chains = 0

        if sim == 'full':
            self._run_first_block(feed_ratios, initiator, rate_matrix, conversion)
        else:
            self._run_abridged(feed_ratios, initiator, rate_matrix, conversion)

        return self.sequences
    
    def run_block_copolymer(self, feed_ratios, initiator_list, rate_matrix, conversion = None):
        num_blocks = feed_ratios.shape[0]
        self.num_monomers = feed_ratios.shape[1]
        self.max_DP = int(np.sum(feed_ratios) + 50)
        self.sequences = np.zeros((self.n_chains, self.max_DP))

        if conversion is None:
            conversion = np.ones((num_blocks, self.num_monomers))

        capped_chains = 0

        for block in range(num_blocks):
            mmol_feed = feed_ratios[block, :]
            initiator = initiator_list[block]
            print("Evaluating 'block' number", block+1)

            if block == 0:
                capped_chains, dead_index = self._run_first_block(mmol_feed, self.num_monomers, initiator, rate_matrix, conversion[block, :])
            else:
                self._add_chains(initiator, capped_chains)
                self._terminate_uncapped(dead_index)
                capped_chains = self._run_block(mmol_feed, initiator, rate_matrix, capped_chains, self.num_monomers, conversion[block,:])
        
        return self.sequences
    
    def run_gradient_copolymer(self, feed_ratios, initiator, rate_matrix):
        num_blocks = feed_ratios.shape[0]
        self.num_monomers = feed_ratios.shape[1]
        self.max_DP = int(np.sum(feed_ratios) + 50)
        self.sequences = np.zeros((self.n_chains, self.max_DP))

        capped_chains = 0

        for block in range(num_blocks):
            mmol_feed = feed_ratios[block, :]
            print("Evaluating 'block' number", block+1)

            if block == 0:
                _, _ = self._run_first_block(mmol_feed, self.num_monomers, initiator, rate_matrix, np.ones(self.num_monomers))
            else:
                self._run_gradient(mmol_feed, rate_matrix, self.num_monomers)
        
        return self.sequences