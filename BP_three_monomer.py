import pandas as pd
from fitting_functions.threemonomer import ThreeMonomerThermalRAFTKineticFitting

exp_data = pd.read_csv('/Users/suprajachittari/Documents/GitHub/prism/sample_data/three_monomer_systems/MPAM-TEGA-OA.csv')
p = ThreeMonomerThermalRAFTKineticFitting(exp_data, 66.36, 42.52, 28.21)
p.extract_rates(1,1,1,1,1,1)