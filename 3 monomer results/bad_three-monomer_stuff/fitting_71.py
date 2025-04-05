import pandas as pd
import numpy as np
from fitting_functions.threemonomer import ThreeMonomerPETRAFTKineticFitting

exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-071_v2.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 41.2, 82.3, 41.3)
rates, m1, m2, m3, conv = p.extract_rates(1,1,1,1,1,1,20.)

np.savetxt("71_converged_rates.csv", rates, delimiter=",", fmt="%f")
np.savetxt("71_conv.csv",conv)
np.savetxt("71_m1.csv",m1)
np.savetxt("71_m2.csv",m2)
np.savetxt("71_m3.csv",m3)