import pandas as pd
import numpy as np
from fitting_functions.threemonomer5 import ThreeMonomerPETRAFTKineticFitting
from matplotlib import pyplot as plt

exp_data = pd.read_excel('sample_data/three_monomer_systems/pdb-5-073_v2.xlsx')
p = ThreeMonomerPETRAFTKineticFitting(exp_data, 18.7, 40.5, 84.9)
bounds = [(0.001,1),(0.001,1),(1,5),(0.001,1),(1,5),(1,5)]
# rates, m1, m2, m3, conv = p.extract_rates(20., bounds)

# np.savetxt("run5_73_rates.csv", rates, delimiter=",", fmt="%f")
# np.savetxt("run5_73_conv.csv",conv)
# np.savetxt("run5_73_m1.csv",m1)
# np.savetxt("run5_73_m2.csv",m2)
# np.savetxt("run5_73_m3.csv",m3)

m1, m2, m3, conv = p.display_overlay([0.7,0.8,2,0.5,2,2], 150)
# m1, m2, m3, conv = p.display_overlay([1,0.8,1.85,0.59,2,2], 60)

plt.plot(conv, m1)
plt.plot(conv, m2)
plt.plot(conv, m3)
plt.scatter(exp_data.iloc[:,0], exp_data.iloc[:,1])
plt.scatter(exp_data.iloc[:,0], exp_data.iloc[:,2])
plt.scatter(exp_data.iloc[:,0], exp_data.iloc[:,3])
plt.show()
