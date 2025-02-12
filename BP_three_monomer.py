import pandas as pd
from fitting_functions.threemonomer import ThreeMonomerThermalRAFTKineticFitting
from fitting_functions.traditional_methods import MeyerLoweryFitting

exp_data = pd.read_csv('/Users/suprajachittari/Documents/GitHub/prism/sample_data/three_monomer_systems/MPAM-TEGA-OA.csv')
p = ThreeMonomerThermalRAFTKineticFitting(exp_data, 66.36, 42.52, 28.21)
p.extract_rates(1,1,1,1,1,1)

# exp_data = pd.read_excel('/Users/suprajachittari/Downloads/meyerlowery_input.xlsx', sheet_name='1-16b')
# p = MeyerLoweryFitting()
# # p.visualize_overlay(exp_data, 2, 2)
# p.extract_rates(exp_data=exp_data)