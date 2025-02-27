import pandas as pd
from fitting_functions.threemonomer import ThreeMonomerThermalRAFTKineticFitting
from fitting_functions.traditional_methods import MeyerLoweryFitting

exp_data = pd.read_excel('/Users/suprajachittari/Documents/GitHub/prism/sample_data/three_monomer_systems/DMA_MPAM_OA.xlsx')
p = ThreeMonomerThermalRAFTKineticFitting(exp_data, 65.25, 60.79, 34.58)
p.extract_rates(1,1,1,1,1,1)

# exp_data = pd.read_excel('/Users/suprajachittari/Downloads/meyerlowery_input.xlsx', sheet_name='1-16b')
# p = MeyerLoweryFitting()
# # p.visualize_overlay(exp_data, 2, 2)
# p.extract_rates(exp_data=exp_data)