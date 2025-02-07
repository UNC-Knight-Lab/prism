import numpy as np
from lmfit import Model
from matplotlib import pyplot as plt

class MeyerLoweryFitting():

    def _meyer_lowery(self, x, fA_0, r_A, r_B):
        gradient_character = ((x / fA_0)**(r_B / (1 - r_B))) * (((1 - x) / (1 - fA_0))**(r_A / (1 - r_A)))
        blocky_character = (((x * (2 - r_A - r_B)) + r_B - 1) / ((fA_0 * (2 - r_A - r_B)) + r_B - 1)) ** (((r_A * r_B) - 1) / ((1 - r_A) * (1 - r_B)))

        return 1 - (gradient_character * blocky_character)
    
    def _recast_data(self, Amol, Bmol):
        fA_0 = Amol[0] / (Amol[0] + Bmol[0])
        fB_0 = Bmol[0] / (Amol[0] + Bmol[0])
        fA = Amol / (Amol + Bmol)
        fB = Bmol / (Amol + Bmol)
        conv = 1 - ((Amol + Bmol) / (Amol[0] + Bmol[0]))

        return fA_0, fA, fB_0, fB, conv

    def _fit(self, fA_0, fA, conv, r_A, r_B):
        fmodel = Model(self._meyer_lowery)
        params = fmodel.make_params(fA_0 = fA_0, 
                                    r_A = {'value':r_A, 'min':0, 'max':5}, 
                                    r_B = {'value':r_B, 'min':0, 'max':5}) # set guesses
        params['fA_0'].vary = False
        result = fmodel.fit(conv, x=fA, params=params, verbose=True)

        return result.params['r_A'], result.params['r_B']

    
    def visualize_overlay(self, exp_data, r_A, r_B):
        Amol = exp_data.iloc[:,1]
        Bmol = exp_data.iloc[:,2]

        fA_0, fA, fB_0, fB, conv = self._recast_data(Amol, Bmol)
        y = self._meyer_lowery(fA, fA_0, r_A, r_B)

        plt.scatter(fA, y)
        plt.scatter(fA, conv)
        plt.ylim([0,1])
        plt.show()

    def extract_rates(self, exp_data, r_A = 0.5, r_B = 0.5):
        Amol = exp_data.iloc[:,1]
        Bmol = exp_data.iloc[:,2]

        fA_0, fA, fB_0, fB, conv = self._recast_data(Amol, Bmol)

        r_A, r_B = self._fit(fA_0, fA, conv, r_A, r_B)
        y = self._meyer_lowery(fA, fA_0, r_A, r_B)

        plt.scatter(fA, y)
        plt.scatter(fA, conv)
        plt.show()

        return r_A, r_B




        