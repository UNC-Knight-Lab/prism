import pandas as pd
import numpy as np
from fitting_functions.threemonomer2 import ThreeMonomerPETRAFTKineticFitting

exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-071_v2.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 41.2, 82.3, 41.3)
bounds = [(0.001,1),(0.001,1),(0.001,5),(1,5),(0.1,5),(0.1,5)]
rates, m1, m2, m3, conv = p.extract_rates(20., bounds)

np.savetxt("de_71_rates_constrained.csv", rates, delimiter=",", fmt="%f")
np.savetxt("de_71_conv_constrained.csv",conv)
np.savetxt("de_71_m1_constrained.csv",m1)
np.savetxt("de_71_m2_constrained.csv",m2)
np.savetxt("de_71_m3_constrained.csv",m3)