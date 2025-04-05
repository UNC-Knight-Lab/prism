import pandas as pd
import numpy as np
from fitting_functions.threemonomer5 import ThreeMonomerPETRAFTKineticFitting

exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-073_v2.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 18.7, 40.5, 84.9)
bounds = [(0.001,1),(0.001,1),(1,5),(0.001,1),(1,5),(1,5)]
rates, m1, m2, m3, conv = p.extract_rates(20., bounds)

np.savetxt("run5_73_rates_constrained_loss.csv", rates, delimiter=",", fmt="%f")
np.savetxt("run5_73_conv_constrained_loss.csv",conv)
np.savetxt("run5_73_m1_constrained_loss.csv",m1)
np.savetxt("run5_73_m2_constrained_loss.csv",m2)
np.savetxt("run5_73_m3_constrained_loss.csv",m3)