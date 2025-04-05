import pandas as pd
import numpy as np
from fitting_functions.threemonomer import ThreeMonomerPETRAFTKineticFitting

exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-073_v2.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 18.7, 40.5, 84.9)
rates, m1, m2, m3, conv = p.extract_rates(1,1,1,1,1,1,20.)

np.savetxt("73_converged_rates.csv", rates, delimiter=",", fmt="%f")
np.savetxt("73_conv.csv",conv)
np.savetxt("73_m1.csv",m1)
np.savetxt("73_m2.csv",m2)
np.savetxt("73_m3.csv",m3)

# p.display_overlay([4.5,1.38,4.47,1.81,4.07,2.95], 20.)
# p.display_overlay([1.44,1.00,3.43,0.79,0.01,2.35], 20.)
# p.test_values(1,1,1,1,1,1,20)

# exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-071_v2.xlsx')
# p = ThreeMonomerPETRAFTKineticFitting(exp_data, 41.2, 82.3, 41.3)
# # p.extract_rates(1,1,1,1,1,1,20.)
# p.display_overlay([4.5,1.38,4.47,1.81,4.07,2.95], 20.)
# # p.display_overlay([1.44,1.00,3.43,0.79,0.01,2.35], 20.)
# # p.test_values(1,1,1,1,1,1,20)