import pandas as pd
from fitting_functions.traditional_methods import MeyerLoweryFitting

exp_data = pd.read_excel('sample_data/two_monomer_systems/PDB-5-063_ML.xlsx')
p = MeyerLoweryFitting()
# p.extract_rates(exp_data, 1.62, 0.44)
p.visualize_overlay(exp_data, 1.62, 0.44)