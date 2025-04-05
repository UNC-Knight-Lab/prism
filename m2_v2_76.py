import pandas as pd
from fitting_functions.ODE_solving import PetRAFTKineticFitting
import numpy as np

exp_data = pd.read_excel('sample_data/two_monomer_systems/pdb-5-076.xlsx')
p = PetRAFTKineticFitting(exp_data, 43.8, 86.7)
rates, convA, convB, tot_conv = p.extract_rates(1., 1.)

np.savetxt("m2_76_rates.csv", rates, delimiter=",", fmt="%f")
np.savetxt("m2_76_conv.csv",tot_conv)
np.savetxt("m2_76_m1.csv",convA)
np.savetxt("m2_76_m2.csv",convB)
