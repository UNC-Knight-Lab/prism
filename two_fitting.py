import pandas as pd
from fitting_functions.ODE_solving import PetRAFTKineticFitting
import numpy as np

exp_data = pd.read_excel('sample_data/two_monomer_systems/pdb-5-070.xlsx')
p = PetRAFTKineticFitting(exp_data, 111.1, 57.7)
rates, convA, convB, tot_conv = p.extract_rates(1., 1.)

np.savetxt("two_monomer_converged_rates.csv", rates, delimiter=",", fmt="%f")
np.savetxt("v2_70_conv.csv",tot_conv)
np.savetxt("v2_70_m1.csv",convA)
np.savetxt("v2_70_m2.csv",convB)
