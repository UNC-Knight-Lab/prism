import pandas as pd
from fitting_functions.threemonomer import ThreeMonomerPETRAFTKineticFitting
from fitting_functions.traditional_methods import MeyerLoweryFitting

exp_data = pd.read_excel('/Users/suprajachittari/Documents/GitHub/prism/sample_data/three_monomer_systems/DMA_MPAM_OA.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 65.25, 60.79, 34.58)
p.extract_rates(1, 1, 1, 1, 1, 1)
# p.display_overlay([2.58563703, 2.83448241, 1.77955357, 3.41667325, 0.43133634, 1.43854887])
# p.test_values(1,1,1,1,1,1)