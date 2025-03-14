from re import L
from unicodedata import digit
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

volume = 1

class ThermalRAFTSequenceEnsemble():

    def __init__(self, n_chains):
        self.n_chains = n_chains
        self.lengths = np.zeros((n_chains), dtype=int)
        self.chain_status = np.zeros((n_chains))

    def _first_monomer(self, mmol_feed, num_monomers, monomer_index):
        u = np.random.random()*np.sum(mmol_feed)

        if u < mmol_feed[0]:
            return monomer_index[0]
        else:
            for i in range(1,num_monomers):
                if u > np.sum(mmol_feed[:i]) and u <= np.sum(mmol_feed[:i+1]):
                    return monomer_index[i]
    
    def _growth_update(self, mmol_feed, new, idx, delta, monomer_index, num_monomers):

        for i in range(num_monomers):
            if new == monomer_index[i] and ((mmol_feed[i] - delta) >= 0):
                self.sequences[idx, self.lengths[idx]] = monomer_index[i]
                mmol_feed[i] -= delta
                self.lengths[idx] += 1
                break
        
        return mmol_feed
    
    def _initiate_with_R(self, mmol_feed, num_monomers, monomer_index, R_mmol, idx, delta):
        new = self._first_monomer(mmol_feed, num_monomers, monomer_index)

        for i in range(num_monomers):
            if new == monomer_index[i] and ((mmol_feed[i] - delta) >= 0):
                self.sequences[idx, self.lengths[idx]] = monomer_index[i]
                mmol_feed[i] -= delta
                R_mmol -= delta
                self.lengths[idx] += 1
                self.chain_status[idx] = 1
                break
        
        return mmol_feed, R_mmol

    def _draw_uninitated_chain(self):
        indices = np.argwhere(self.chain_status == 0)

        if indices.shape[0] == 0:
            return False
        else:
            return random.choice(indices)
    
    def _draw_uncapped_chain(self):
        indices = np.argwhere(self.chain_status == 1)

        return random.choice(indices)

    def _draw_capped_chain(self):
        indices = np.argwhere(self.chain_status == 2)

        return random.choice(indices)

    def _growth_move(self, mmol_feed, num_monomers, monomer_index, last_monomer, r_matrix, capped_chains, CTA_mmol, R_mmol):
        rates_add = np.zeros((num_monomers + 3,3)) # first column is normalized rate, second is addition, third is sum

        for i in range(num_monomers):
            if last_monomer == monomer_index[i]:
                for j in range(num_monomers):
                    rates_add[j,0] = r_matrix[i,j] * mmol_feed[j]
                    rates_add[j,1] = monomer_index[j]
                    rates_add[j,2] = sum(rates_add[:,0])

        k_cap = 100

        # chain caps with another chain
        rates_add[num_monomers,0] = capped_chains*k_cap
        rates_add[num_monomers,1] = num_monomers + 1
        rates_add[num_monomers,2] = sum(rates_add[:,0])

        # chain caps with CTA
        rates_add[num_monomers+1,0] = CTA_mmol*k_cap*100
        rates_add[num_monomers+1,1] = num_monomers + 2
        rates_add[num_monomers+1,2] = sum(rates_add[:,0])

        # chain terminates with R group
        rates_add[num_monomers+2,0] = R_mmol*1. #/ norm_rates
        rates_add[num_monomers+2,1] = num_monomers + 3
        rates_add[num_monomers+2,2] = sum(rates_add[:,0])

        # random number generation for move
        u = np.random.random()*rates_add[-1,2]

        if u <= rates_add[0, 2]:
            return rates_add[0,1]
        else:
            for i in range(1,num_monomers + 3):
                if rates_add[i-1,2] < u <= rates_add[i,2]:
                    return rates_add[i,1]
    
    def _capping_update(self, chain, capping_index):
        self.sequences[chain, self.lengths[chain]] = capping_index
        self.lengths[chain] += 1
        self.chain_status[chain] = 2
    
    def _uncapping_update(self, chain, uncapped_index):
        self.sequences[chain, self.lengths[chain] - 1] = uncapped_index
        self.lengths[chain] -= 1
        self.chain_status[chain] = 1
    
    def _first_capping(self, idx, CTA_mmol, R_mmol, delta, capped_chains, capped_index):
        if CTA_mmol - delta > 0:
            self.sequences[idx, self.lengths[idx]] = capped_index
            R_mmol += delta
            CTA_mmol -= delta
            self.lengths[idx] += 1
            self.chain_status[idx] = 2
            capped_chains += 1
        
            # print(self.sequences[idx,:])
        return CTA_mmol, R_mmol, delta, capped_chains
    
    def _chain_termination(self, chain, R_mmol, delta, dead_index):
        if R_mmol - delta > 0:
            self.sequences[chain, self.lengths[chain]] = dead_index
            R_mmol += delta
            self.lengths[chain] += 1
            self.chain_status[chain] = 3
        
        return R_mmol
    
    def _force_growth(self, chain, mmol_feed, r_matrix, num_monomers, monomer_index):
        last_monomer = self.sequences[chain, (self.lengths[chain] - 1)]

        rates_add = np.zeros((num_monomers,3)) # first column is normalized rate, second is addition, third is sum

        for i in range(num_monomers):
            if last_monomer == monomer_index[i]:
                for j in range(num_monomers):
                    rates_add[j,0] = r_matrix[i,j] * mmol_feed[j]
                    rates_add[j,1] = monomer_index[j]
                    rates_add[j,2] = sum(rates_add[:,0])
        
        # random number generation for move
        u = np.random.random()*rates_add[-1,2]

        if u <= rates_add[0, 2]:
            return rates_add[0,1]
        else:
            for i in range(1,num_monomers + 3):
                if rates_add[i-1,2] < u <= rates_add[i,2]:
                    return rates_add[i,1]

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


    def _run_first_block(self, mmol_feed, num_monomers, inits, r_matrix, conversion):
        delta = 1 / self.n_chains
        num_initiated_chains = int(inits * self.n_chains)
        CTA_mmol = 1.
        R_mmol = 0.
        capped_chains = 0.

        uncapped_index = 0
        monomer_indexes = np.arange(1, num_monomers+1)
        capped_index = num_monomers + 1
        dead_index = capped_index + 1

        left_over = (1 - conversion) * mmol_feed
        
        for i in range(num_initiated_chains):
            new = self._first_monomer(mmol_feed, num_monomers, monomer_indexes)
            mmol_feed = self._growth_update(mmol_feed, new, i, delta, monomer_indexes, num_monomers)
            
        self.chain_status[0:num_initiated_chains] = 1

        attempt = 1

        while capped_chains <= 0.98*self.n_chains:

            while (R_mmol - delta) >= 0:
                chain = self._draw_uninitated_chain()
                if chain == False:
                    break
                else:
                    mmol_feed, R_mmol = self._initiate_with_R(mmol_feed, num_monomers, monomer_indexes, R_mmol, chain, delta)
            
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 0:
                mmol_feed, R_mmol = self._initiate_with_R(mmol_feed, num_monomers, monomer_indexes, R_mmol, chain, delta)
            elif self.chain_status[chain] == 1:
                last_monomer = self.sequences[chain, (self.lengths[chain] - 1)]

                new = self._growth_move(mmol_feed, num_monomers, monomer_indexes, last_monomer, r_matrix, capped_chains, CTA_mmol, R_mmol)

                if new == num_monomers + 1: # chain caps with another chain
                    self._capping_update(chain, capped_index)
                    swap_chain = self._draw_capped_chain()
                    self._uncapping_update(swap_chain, uncapped_index)

                elif new == num_monomers + 2: # chain caps with CTA
                    CTA_mmol, R_mmol, delta, capped_chains = self._first_capping(chain, CTA_mmol, R_mmol, delta, capped_chains, capped_index)
                
                elif new == num_monomers + 3: # chain terminates
                    R_mmol = self._chain_termination(chain, R_mmol, delta, dead_index)
                
                else: # growth move
                    mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

            elif self.chain_status[chain] == 2: # capped chain uncaps with another chain
                self._uncapping_update(chain, uncapped_index)
                swap_chain = self._draw_uncapped_chain()
                self._capping_update(swap_chain, capped_index)
        
            attempt += 1
        
        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 1:
                new = self._force_growth(chain, mmol_feed, r_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)
                self._capping_update(chain, capped_index)
                swap_chain = self._draw_capped_chain()
                self._uncapping_update(swap_chain, uncapped_index)
            else:
                self._uncapping_update(chain, uncapped_index)
                new = self._force_growth(chain, mmol_feed, r_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

                swap_chain = self._draw_uncapped_chain()
                self._capping_update(swap_chain, capped_index)
            
            result = (mmol_feed - left_over) <= delta
            # print(mmol_feed, left_over)
            # print(result)

            if result.all() == True:
                break
            
            # print(mmol_feed)

        return capped_chains, dead_index

    def _run_block(self, mmol_feed, inits, rate_matrix, capped_chains, num_monomers, conversion):
        delta = 1.0 / capped_chains
        num_initiated_chains = int(inits * capped_chains)

        uncapped_index = 0
        monomer_indexes = np.arange(1, num_monomers+1)
        capped_index = num_monomers + 1

        left_over = (1 - conversion) * mmol_feed

        for i in range(num_initiated_chains):
            new = self._first_monomer(mmol_feed, num_monomers, monomer_indexes)
            mmol_feed = self._growth_update(mmol_feed, new, i, delta, monomer_indexes, num_monomers)
        
        self.chain_status[0:num_initiated_chains] = 1

        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 1:
                new = self._force_growth(chain, mmol_feed, rate_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)
                self._capping_update(chain, capped_index)
                swap_chain = self._draw_capped_chain()
                self._uncapping_update(swap_chain, uncapped_index)
            else:
                self._uncapping_update(chain, uncapped_index)
                new = self._force_growth(chain, mmol_feed, rate_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

                swap_chain = self._draw_uncapped_chain()
                self._capping_update(swap_chain, capped_index)
            
            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break
        
        return capped_chains
    
    def _run_gradient(self, mmol_feed, rate_matrix, num_monomers):
        delta = 1.0 / self.n_chains

        uncapped_index = 0
        monomer_indexes = np.arange(1, num_monomers+1)
        capped_index = num_monomers + 1

        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 1:
                new = self._force_growth(chain, mmol_feed, rate_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)
                self._capping_update(chain, capped_index)
                swap_chain = self._draw_capped_chain()
                self._uncapping_update(swap_chain, uncapped_index)
            else:
                self._uncapping_update(chain, uncapped_index)
                new = self._force_growth(chain, mmol_feed, rate_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

                swap_chain = self._draw_uncapped_chain()
                self._capping_update(swap_chain, capped_index)
    
            if (np.any(mmol_feed != 0) == True) & ((mmol_feed <= delta).all() == True):
                break

    
    def run_statistical(self, feed_ratios, initiator, rate_matrix, conversion = None):
        num_monomers = feed_ratios.shape[0]
        self.max_DP = int(np.sum(feed_ratios) + 50)
        self.sequences = np.zeros((self.n_chains, self.max_DP))

        if conversion is None:
            conversion = np.ones(num_monomers)

        capped_chains = 0

        capped_chains, dead_index = self._run_first_block(feed_ratios, num_monomers, initiator, rate_matrix, conversion)

        return self.sequences

    def run_block_copolymer(self, feed_ratios, initiator_list, rate_matrix, conversion = None):
        num_blocks = feed_ratios.shape[0]
        num_monomers = feed_ratios.shape[1]
        self.max_DP = int(np.sum(feed_ratios) + 50)
        self.sequences = np.zeros((self.n_chains, self.max_DP))

        if conversion is None:
            conversion = np.ones((num_blocks, num_monomers))

        capped_chains = 0

        for block in range(num_blocks):
            mmol_feed = feed_ratios[block, :]
            initiator = initiator_list[block]
            print("Evaluating 'block' number", block+1)

            if block == 0:
                capped_chains, dead_index = self._run_first_block(mmol_feed, num_monomers, initiator, rate_matrix, conversion[block, :])
            else:
                self._add_chains(initiator, capped_chains)
                self._terminate_uncapped(dead_index)
                capped_chains = self._run_block(mmol_feed, initiator, rate_matrix, capped_chains, num_monomers, conversion[block,:])
        
        return self.sequences
    
    def run_gradient_copolymer(self, feed_ratios, initiator, rate_matrix):
        num_blocks = feed_ratios.shape[0]
        num_monomers = feed_ratios.shape[1]
        self.max_DP = int(np.sum(feed_ratios) + 50)
        self.sequences = np.zeros((self.n_chains, self.max_DP))

        capped_chains = 0

        for block in range(num_blocks):
            mmol_feed = feed_ratios[block, :]
            print("Evaluating 'block' number", block+1)

            if block == 0:
                _, _ = self._run_first_block(mmol_feed, num_monomers, initiator, rate_matrix, np.ones(num_monomers))
            else:
                self._run_gradient(mmol_feed, rate_matrix, num_monomers)
        
        return self.sequences

class PETRAFTSequenceEnsemble():

    def __init__(self, n_chains):
        self.n_chains = n_chains
        self.lengths = np.zeros((n_chains), dtype=int)
        self.chain_status = np.zeros((n_chains))

    def _first_monomer(self, mmol_feed, num_monomers, monomer_index):
        u = np.random.random()*np.sum(mmol_feed)

        if u < mmol_feed[0]:
            return monomer_index[0]
        else:
            for i in range(1,num_monomers):
                if u > np.sum(mmol_feed[:i]) and u <= np.sum(mmol_feed[:i+1]):
                    return monomer_index[i]
    
    def _growth_update(self, mmol_feed, new, idx, delta, monomer_index, num_monomers):

        for i in range(num_monomers):
            if new == monomer_index[i] and ((mmol_feed[i] - delta) >= 0):
                self.sequences[idx, self.lengths[idx]] = monomer_index[i]
                mmol_feed[i] -= delta
                self.lengths[idx] += 1
                break
        
        return mmol_feed
    
    def _draw_uncapped_chain(self):
        indices = np.argwhere(self.chain_status == 1)

        return random.choice(indices)

    def _draw_capped_chain(self):
        indices = np.argwhere(self.chain_status == 2)

        return random.choice(indices)

    def _growth_move(self, mmol_feed, num_monomers, monomer_index, last_monomer, r_matrix, capped_chains, R_mmol, Z_mmol, delta):
        rates_add = np.zeros((num_monomers + 3,3)) # first column is normalized rate, second is addition, third is sum

        for i in range(num_monomers):
            if last_monomer == monomer_index[i]:
                for j in range(num_monomers):
                    rates_add[j,0] = r_matrix[i,j] * mmol_feed[j]
                    rates_add[j,1] = monomer_index[j]
                    rates_add[j,2] = sum(rates_add[:,0])

        k_cap = 100
        k_terminate = 1

        # chain caps with another chain
        rates_add[num_monomers,0] = (capped_chains*delta)*k_cap
        rates_add[num_monomers,1] = num_monomers + 1
        rates_add[num_monomers,2] = sum(rates_add[:,0])

        # chain caps with Z_group
        rates_add[num_monomers+1,0] = Z_mmol*k_cap*100
        rates_add[num_monomers+1,1] = num_monomers + 2
        rates_add[num_monomers+1,2] = sum(rates_add[:,0])

        # chain terminates with R group
        rates_add[num_monomers+2,0] = R_mmol*k_terminate
        rates_add[num_monomers+2,1] = num_monomers + 3
        rates_add[num_monomers+2,2] = sum(rates_add[:,0])

        # random number generation for move
        u = np.random.random()*rates_add[-1,2]
        # print(rates_add)

        if u <= rates_add[0, 2]:
            return rates_add[0,1]
        else:
            for i in range(1,num_monomers + 3):
                if rates_add[i-1,2] < u <= rates_add[i,2]:
                    return rates_add[i,1]
    
    def _capping_update(self, chain, capping_index):
        self.sequences[chain, self.lengths[chain]] = capping_index
        self.lengths[chain] += 1
        self.chain_status[chain] = 2
    
    def _uncapping_update(self, chain, uncapped_index):
        self.sequences[chain, self.lengths[chain] - 1] = uncapped_index
        self.lengths[chain] -= 1
        self.chain_status[chain] = 1
    
    def _capping_no_exchange(self, idx, Z_mmol, delta, capped_chains, capped_index):
        self.sequences[idx, self.lengths[idx]] = capped_index
        Z_mmol -= delta
        self.lengths[idx] += 1
        self.chain_status[idx] = 2
        capped_chains += 1

        return Z_mmol, capped_chains

    def _uncapping_move(self, pc_mmol, capped_chains, delta): # chain uncaps with other chain or through photocatalyst
        k_cap = 100

        num_uncapped = self.n_chains - capped_chains

        rates_cap = np.zeros((2,2))
        
        # chain uncaps with another chain
        rates_cap[0,0] = k_cap*(num_uncapped*delta)
        rates_cap[0,1] = sum(rates_cap[:,0])

        # chain uncaps with photocatalyst
        rates_cap[1,0] = k_cap*pc_mmol
        rates_cap[1,1] = sum(rates_cap[:,0])
        # print(rates_cap)

        u = np.random.random()*rates_cap[-1,1]

        if u < rates_cap[0,1]:
            return 'chain'
        else:
            return 'photocatalyst'
    
    def _capping_type(self, Z_mmol, capped_chains, delta):
        k_cap = 100

        rates_cap = np.zeros((2,2))
        
        # chain caps with another chain
        rates_cap[0,0] = k_cap*(capped_chains*delta)
        rates_cap[0,1] = sum(rates_cap[:,0])

        # chain caps with Z group
        rates_cap[1,0] = k_cap*Z_mmol*100
        rates_cap[1,1] = sum(rates_cap[:,0])

        u = np.random.random()*rates_cap[-1,1]

        if u < rates_cap[0,1]:
            return 'chain'
        else:
            return 'photocatalyst'
       
    def _chain_termination(self, chain, R_mmol, delta, dead_index):
        if R_mmol - delta > 0:
            self.sequences[chain, self.lengths[chain]] = dead_index
            R_mmol += delta
            self.lengths[chain] += 1
            self.chain_status[chain] = 3
        
        return R_mmol
    
    def _force_growth(self, chain, mmol_feed, r_matrix, num_monomers, monomer_index):
        last_monomer = self.sequences[chain, (self.lengths[chain] - 1)]

        rates_add = np.zeros((num_monomers,3)) # first column is normalized rate, second is addition, third is sum

        for i in range(num_monomers):
            if last_monomer == monomer_index[i]:
                for j in range(num_monomers):
                    rates_add[j,0] = r_matrix[i,j] * mmol_feed[j]
                    rates_add[j,1] = monomer_index[j]
                    rates_add[j,2] = sum(rates_add[:,0])
        
        # random number generation for move
        u = np.random.random()*rates_add[-1,2]

        if u <= rates_add[0, 2]:
            return rates_add[0,1]
        else:
            for i in range(1,num_monomers + 3):
                if rates_add[i-1,2] < u <= rates_add[i,2]:
                    return rates_add[i,1]

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

    def _run_first_block(self, mmol_feed, num_monomers, pc, r_matrix, conversion):
        delta = 1 / self.n_chains
        pc_mmol = 0.5*pc # excited radical species
        capped_chains = 0.

        uncapped_index = 0
        monomer_indexes = np.arange(1, num_monomers+1)
        capped_index = num_monomers + 1
        dead_index = capped_index + 1

        left_over = (1 - conversion) * mmol_feed
        
        for i in range(self.n_chains):
            new = self._first_monomer(mmol_feed, num_monomers, monomer_indexes)
            mmol_feed = self._growth_update(mmol_feed, new, i, delta, monomer_indexes, num_monomers)
            
        self.chain_status[:] = 1
        R_mmol = 1.
        Z_mmol = 1.

        while capped_chains <= 0.9*self.n_chains:

            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 1:
                last_monomer = self.sequences[chain, (self.lengths[chain] - 1)]

                new = self._growth_move(mmol_feed, num_monomers, monomer_indexes, last_monomer, r_matrix, capped_chains, R_mmol, Z_mmol, delta)

                if new == num_monomers + 1: # chain caps with another chain
                    self._capping_update(chain, capped_index)
                    swap_chain = self._draw_capped_chain()
                    self._uncapping_update(swap_chain, uncapped_index)

                elif new == num_monomers + 2: # chain caps with photocatalyst mechanism
                    Z_mmol, capped_chains = self._capping_no_exchange(chain, Z_mmol, delta, capped_chains, capped_index)
                    # print("capping by PC")
                elif new == num_monomers + 3: # chain terminates
                    R_mmol = self._chain_termination(chain, R_mmol, delta, dead_index)
                
                else: # growth move
                    mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

            elif self.chain_status[chain] == 2: # capped chain uncaps
                uncapping_type = self._uncapping_move(pc_mmol, capped_chains, delta)
                # print(uncapping_type)
                if uncapping_type == 'chain': # uncaps with another chain
                    self._uncapping_update(chain, uncapped_index)
                    swap_chain = self._draw_uncapped_chain()
                    self._capping_update(swap_chain, capped_index)
                else: # uncaps with Z group
                    self._uncapping_update(chain, uncapped_index)
                    Z_mmol += delta
                    capped_chains -= 1

        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 1: # chain is uncapped
                new = self._force_growth(chain, mmol_feed, r_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

                capping_type = self._capping_type(Z_mmol, capped_chains, delta)
                
                if capping_type == 'chain': # caps with another chain

                    self._capping_update(chain, capped_index)
                    swap_chain = self._draw_capped_chain()
                    self._uncapping_update(swap_chain, uncapped_index)

                else: # caps with Z group
                    Z_mmol, capped_chains = self._capping_no_exchange(chain, Z_mmol, delta, capped_chains, capped_index)
            else: # chain is capped
                uncapping_type = self._uncapping_move(pc_mmol, capped_chains, delta)

                if uncapping_type == 'chain': # uncaps with another chain
                    self._uncapping_update(chain, uncapped_index)
                    new = self._force_growth(chain, mmol_feed, r_matrix, num_monomers, monomer_indexes)
                    mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

                    swap_chain = self._draw_uncapped_chain()
                    self._capping_update(swap_chain, capped_index)
                else: # uncaps with Z group
                    self._uncapping_update(chain, uncapped_index)
                    Z_mmol += delta
                    capped_chains -= 1
            # print(mmol_feed)
                                
            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break

        return capped_chains, dead_index

    def _run_block(self, mmol_feed, pc_mmol, r_matrix, capped_chains, num_monomers, conversion):
        delta = 1.0 / capped_chains

        uncapped_index = 0
        monomer_indexes = np.arange(1, num_monomers+1)
        capped_index = num_monomers + 1
        Z_mmol = 0.
        pc_mmol *= 0.5

        left_over = (1 - conversion) * mmol_feed

        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 1: # chain is uncapped
                new = self._force_growth(chain, mmol_feed, r_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

                capping_type = self._capping_type(Z_mmol, capped_chains, delta)
                
                if capping_type == 'chain': # caps with another chain

                    self._capping_update(chain, capped_index)
                    swap_chain = self._draw_capped_chain()
                    self._uncapping_update(swap_chain, uncapped_index)

                else: # caps with Z group
                    Z_mmol, capped_chains = self._capping_no_exchange(chain, Z_mmol, delta, capped_chains, capped_index)
            else: # chain is capped
                uncapping_type = self._uncapping_move(pc_mmol, capped_chains, delta)

                if uncapping_type == 'chain': # uncaps with another chain
                    self._uncapping_update(chain, uncapped_index)
                    new = self._force_growth(chain, mmol_feed, r_matrix, num_monomers, monomer_indexes)
                    mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

                    swap_chain = self._draw_uncapped_chain()
                    self._capping_update(swap_chain, capped_index)
                else: # uncaps with Z group
                    self._uncapping_update(chain, uncapped_index)
                    Z_mmol += delta
                    capped_chains -= 1

                                
            result = (mmol_feed - left_over) <= delta

            if result.all() == True:
                break

        
        return capped_chains
    
    def _run_gradient(self, mmol_feed, rate_matrix, num_monomers):
        delta = 1.0 / self.n_chains

        uncapped_index = 0
        monomer_indexes = np.arange(1, num_monomers+1)
        capped_index = num_monomers + 1

        while np.max(self.lengths) <= self.max_DP:
            chain = random.choice(np.arange(0,self.n_chains))

            if self.chain_status[chain] == 1:
                new = self._force_growth(chain, mmol_feed, rate_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)
                self._capping_update(chain, capped_index)
                swap_chain = self._draw_capped_chain()
                self._uncapping_update(swap_chain, uncapped_index)
            else:
                self._uncapping_update(chain, uncapped_index)
                new = self._force_growth(chain, mmol_feed, rate_matrix, num_monomers, monomer_indexes)
                mmol_feed = self._growth_update(mmol_feed, new, chain, delta, monomer_indexes, num_monomers)

                swap_chain = self._draw_uncapped_chain()
                self._capping_update(swap_chain, capped_index)
    
            if (np.any(mmol_feed != 0) == True) & ((mmol_feed <= delta).all() == True):
                break

    
    def run_statistical(self, feed_ratios, photocatalyst, rate_matrix, conversion = None):
        num_monomers = feed_ratios.shape[0]
        self.max_DP = int(np.sum(feed_ratios) + 50)
        self.sequences = np.zeros((self.n_chains, self.max_DP))

        if conversion is None:
            conversion = np.ones(num_monomers)

        capped_chains = 0

        capped_chains, dead_index = self._run_first_block(feed_ratios, num_monomers, photocatalyst, rate_matrix, conversion)

        return self.sequences

    def run_block_copolymer(self, feed_ratios, pc_mmol, rate_matrix, conversion = None):
        num_blocks = feed_ratios.shape[0]
        num_monomers = feed_ratios.shape[1]
        self.max_DP = int(np.sum(feed_ratios) + 50)
        self.sequences = np.zeros((self.n_chains, self.max_DP))

        if conversion is None:
            conversion = np.ones((num_blocks, num_monomers))

        capped_chains = 0

        for block in range(num_blocks):
            mmol_feed = feed_ratios[block, :]
            print("Evaluating 'block' number", block+1)

            if block == 0:
                capped_chains, dead_index = self._run_first_block(mmol_feed, num_monomers, pc_mmol, rate_matrix, conversion[block, :])
            else:
                self._terminate_uncapped(dead_index)
                capped_chains = self._run_block(mmol_feed, pc_mmol, rate_matrix, capped_chains, num_monomers, conversion[block,:])
        
        return self.sequences
    
    # def run_gradient_copolymer(self, feed_ratios, initiator, rate_matrix):
    #     num_blocks = feed_ratios.shape[0]
    #     num_monomers = feed_ratios.shape[1]
    #     self.max_DP = int(np.sum(feed_ratios) + 50)
    #     self.sequences = np.zeros((self.n_chains, self.max_DP))

    #     capped_chains = 0

    #     for block in range(num_blocks):
    #         mmol_feed = feed_ratios[block, :]
    #         print("Evaluating 'block' number", block+1)

    #         if block == 0:
    #             _, _ = self._run_first_block(mmol_feed, num_monomers, initiator, rate_matrix, np.ones(num_monomers))
    #         else:
    #             self._run_gradient(mmol_feed, rate_matrix, num_monomers)
        
    #     return self.sequences

