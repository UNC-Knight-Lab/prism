import pandas as pd
from fitting_functions.threemonomer import ThreeMonomerPETRAFTKineticFitting

exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-071.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 41.16, 82.32, 43.15)
p.extract_rates(1,1,1,1,1,1)
# p.display_overlay([1,1,1,1,1,1])
# p.test_values(1,1,1,1,1,1)


# exp_data = pd.read_excel('/Users/suprajachittari/Documents/GitHub/prism/sample_data/two_monomer_systems/MPAM_OA_50_50.xlsx')
# p = PetRAFTKineticFitting(exp_data, 56.39, 59.94)
# # p.extract_rates(0.43777963, 10.37974691)
# p.display_overlay([0.43081429, 3.91365678])

# exp_data = pd.read_excel('/Users/suprajachittari/Documents/GitHub/prism/sample_data/two_monomer_systems/MPAM_OA_50_50_ML.xlsx')
# p = MeyerLoweryFitting()
# p.visualize_overlay(exp_data, 0.43, 10.0)
# p.extract_rates(exp_data=exp_data)