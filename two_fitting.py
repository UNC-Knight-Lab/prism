import pandas as pd
from fitting_functions.ODE_solving import PetRAFTKineticFitting

exp_data = pd.read_excel('sample_data/two_monomer_systems/pdb-5-070.xlsx')
p = PetRAFTKineticFitting(exp_data, 111.1, 57.7)
p.extract_rates(1., 1.)
