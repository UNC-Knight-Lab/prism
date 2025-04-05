import pandas as pd
import numpy as np
from fitting_functions.threemonomer5 import ThreeMonomerPETRAFTKineticFitting
from matplotlib import pyplot as plt

exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-073_v2_clipped.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 18.7, 40.5, 84.9)
bounds = [(0.001,1),(0.001,1),(1,5),(0.001,1),(1,5),(1,5)]
guess_ = np.array([0.9,0.8,2,0.5,2,2])
rates, m1, m2, m3, conv = p.extract_rates(20., bounds, guess_)

# np.savetxt("v6_73_rates.csv", rates, delimiter=",", fmt="%f")
# np.savetxt("v6_73_conv.csv",conv)
# np.savetxt("v6_73_m1.csv",m1)
# np.savetxt("v6_73_m2.csv",m2)
# np.savetxt("v6_73_m3.csv",m3)

# m1, m2, m3, conv = p.display_overlay([0.28,1.0,1,0.8,5,3.5], 150)
# m1, m2, m3, conv = p.display_overlay([1,0.8,1.87,0.59,1.96,2.02], 150)

# plt.plot(conv, m1)
# plt.plot(conv, m2)
# plt.plot(conv, m3)
# plt.scatter(exp_data.iloc[:,0], exp_data.iloc[:,1])
# plt.scatter(exp_data.iloc[:,0], exp_data.iloc[:,2])
# plt.scatter(exp_data.iloc[:,0], exp_data.iloc[:,3])
# plt.show()