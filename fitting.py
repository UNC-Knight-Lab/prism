import pandas as pd
from fitting_functions.threemonomer import ThreeMonomerPETRAFTKineticFitting
from fitting_functions.ODE_solving import PetRAFTKineticFitting
from fitting_functions.traditional_methods import MeyerLoweryFitting

exp_data = pd.read_excel('/Users/suprajachittari/Documents/GitHub/prism/sample_data/three_monomer_systems/DMA_MPAM_OA.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 65.25, 60.79, 34.58)
p.extract_rates(1,1,1,1,1,1)
# p.display_overlay([1.02836752, 0.01, 0.32585111, 1.30661458, 0.12827558, 0.15034467])


# exp_data = pd.read_excel('/Users/suprajachittari/Documents/GitHub/prism/sample_data/two_monomer_systems/MPAM_OA_50_50.xlsx')
# p = PetRAFTKineticFitting(exp_data, 56.39, 59.94)
# # p.extract_rates(0.43777963, 10.37974691)
# p.display_overlay([0.43081429, 3.91365678])

# exp_data = pd.read_excel('/Users/suprajachittari/Documents/GitHub/prism/sample_data/two_monomer_systems/MPAM_OA_50_50_ML.xlsx')
# p = MeyerLoweryFitting()
# p.visualize_overlay(exp_data, 0.43, 10.0)
# p.extract_rates(exp_data=exp_data)