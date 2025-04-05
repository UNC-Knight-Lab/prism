import pandas as pd
from fitting_functions.ODE_solving import PetRAFTKineticFitting
import numpy as np

exp_data = pd.read_excel('sample_data/two_monomer_systems/pdb-5-063.xlsx')
p = PetRAFTKineticFitting(exp_data, 52.3, 98.4)
rates, convA, convB, tot_conv = p.extract_rates(1., 1.)

np.savetxt("m2_v2_63_rates.csv", rates, delimiter=",", fmt="%f")
np.savetxt("m2_v2_63_conv.csv",tot_conv)
np.savetxt("m2_v2_63_m1.csv",convA)
np.savetxt("m2_v2_63_m2.csv",convB)
