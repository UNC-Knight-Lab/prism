import pandas as pd
import numpy as np
from fitting_functions.threemonomer2 import ThreeMonomerPETRAFTKineticFitting

exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-073_v2.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 18.7, 40.5, 84.9)
bounds = [(0.001,1),(0.001,1),(0.001,5),(1,5),(0.1,5),(0.1,5)]
rates, m1, m2, m3, conv = p.extract_rates(20., bounds)

np.savetxt("de_73_rates_constrained.csv", rates, delimiter=",", fmt="%f")
np.savetxt("de_73_conv_constrained.csv",conv)
np.savetxt("de_73_m1_constrained.csv",m1)
np.savetxt("de_73_m2_constrained.csv",m2)
np.savetxt("de_73_m3_constrained.csv",m3)