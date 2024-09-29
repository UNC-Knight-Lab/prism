import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.optimize import least_squares, minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.evaluator import Evaluator
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.core.population import Population, Individual

k_s = 100
k_j = 100
k_c = 100
k_d = 0.1

exp_data = pd.read_csv('/Users/suprajachittari/Documents/peter/sequence/MPB_tba_oa_DMA.csv')

class PetRAFTKineticFitting():
    def __init__(self, exp_data, A_mol, B_mol, C_mol):
        self.exp_data = exp_data
        self.A_mol = A_mol
        self.B_mol = B_mol
        self.C_mol = C_mol
    
    def _ODE(self, t, x, k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d):
        i, i_r, cta, R_r, a, b, c, x_a, x_b, x_c, x_ac, x_bc, x_cc, d = x

        dxdt = np.zeros((14))

        dxdt[0] = -k_s*i
        dxdt[1] = k_s*i - k_j*i_r*(a + b + c) - k_d*i_r*(x_a + x_b + x_c)
        dxdt[2] = -k_c*cta*(x_a + x_b + x_c)
        dxdt[3] = -k_j*R_r*(a + b + c) + k_c*cta*(x_a + x_b + x_c) - k_d*R_r*(x_a + x_b + x_c)
        dxdt[4] = -k_d*i_r*a -k_j*R_r*a - k_AA*x_a*a - k_BA*x_b*a - k_CA*x_c*a
        dxdt[5] = -k_d*i_r*b -k_j*R_r*b - k_AB*x_a*b - k_BB*x_b*b - k_CB*x_c*b
        dxdt[6] = -k_d*i_r*c -k_j*R_r*c - k_AC*x_a*c - k_BC*x_b*c - k_CC*x_c*c
        dxdt[7] = k_j*i_r*a + k_j*R_r*a + k_BA*x_b*a - k_AB*x_a*b - k_AC*x_a*c + k_CA*x_c*a - k_c*x_a*cta - k_c*x_a*x_bc + k_c*x_ac*x_b + k_c*x_ac*x_c - k_c*x_cc*x_a - k_d*R_r*x_a - k_d*i_r*x_a
        dxdt[8] = k_j*i_r*b + k_j*R_r*b - k_BA*x_b*a + k_AB*x_a*b - k_BC*x_b*c + k_CB*x_c*b - k_c*x_b*cta + k_c*x_a*x_bc - k_c*x_ac*x_b + k_c*x_bc*x_c - k_c*x_cc*x_b - k_d*R_r*x_b - k_d*i_r*x_b
        dxdt[9] = k_j*i_r*c + k_j*R_r*c + k_AC*x_a*c + k_BC*x_b*c - k_CA*x_c*a - k_CB*x_c*b - k_c*x_c*cta - k_c*x_bc*x_c + k_c*x_cc*x_b - k_c*x_ac*x_c + k_c*x_cc*x_a - k_d*R_r*x_c - k_d*i_r*x_c
        dxdt[10] = k_c*x_a*cta - k_c*x_ac*x_b + k_c*x_bc*x_a - k_c*x_ac*c + k_c*x_cc*x_a
        dxdt[11] = k_c*x_b*cta - k_c*x_bc*x_c + k_c*x_cc*x_b + k_c*x_ac*x_b - k_c*x_bc*x_a
        dxdt[12] = k_c*x_c*cta + k_c*x_bc*x_c - k_c*x_cc*x_b + k_c*x_ac*x_c - k_c*x_cc*x_a
        dxdt[13] = k_d*i_r*(x_a + x_b + x_c) + k_d*R_r*(x_a + x_b + x_c)

        return dxdt
    
    def _integrate_ODE(self, k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d):
        # initial condition
        x0 = np.zeros((14))
        x0[0] = 0.05
        x0[2] = 1.
        x0[4] = self.A_mol
        x0[5] = self.B_mol
        x0[6] = self.C_mol

        # parameters
        param_tuple = (k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        t_span = (0, 1000.0)
        t_eval = np.linspace(0, 1000., 500)

        sol = solve_ivp(self._ODE, t_span, x0, args=param_tuple, t_eval=t_eval)

        return sol
    
    def _convert_XF(self, sol):
        f_iA = self.A_mol / (self.A_mol + self.B_mol + self.C_mol)
        f_iB = self.B_mol / (self.A_mol + self.B_mol + self.C_mol)
        f_iC = self.C_mol / (self.A_mol + self.B_mol + self.C_mol)
    
        A_conc = sol.y[4]
        B_conc = sol.y[5]
        C_conc = sol.y[6]

        f_A = (A_conc / self.A_mol) * f_iA
        f_B = (B_conc / self.B_mol) * f_iB
        f_C = (C_conc / self.C_mol) * f_iC

        f_A[0] = f_iA
        f_B[0] = f_iB
        f_C[0] = f_iC

        fracA = f_A / (f_A + f_B + f_C)
        fracB = f_B / (f_A + f_B + f_C)
        totalfrac = ((f_iA - f_A) + (f_iB - f_B) + (f_iC - f_C)) / (f_iA + f_iB + f_iC)
        print(fracA)
        idx = np.argmax(totalfrac > 0.96)

        return fracA[:idx], fracB[:idx], totalfrac[:idx]
    
    def _sum_square_residuals(self, pred_X, pred_F, i):
        interpolator = interp1d(pred_X, pred_F, kind='linear')
        y_interpolated = interpolator(self.exp_data.iloc[:,0])

        residuals = self.exp_data.iloc[:,i] - y_interpolated

        return np.sum(residuals**2)
    
    def evaluate(self, k):
        k_AB, k_AC, k_BA, k_BC, k_CA, k_CB = k
        k_AA = 1.
        k_BB = 1.
        k_CC = 1.

        sol = self._integrate_ODE(k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        pred_F1, pred_F2, pred_X = self._convert_XF(sol)
        print(pred_F1, pred_F2, pred_X)
        loss1 = self._sum_square_residuals(pred_X, pred_F1, 1)
        loss2 = self._sum_square_residuals(pred_X, pred_F2, 2)

        return loss1, loss2
    
    def display_overlay(self, new_k):
        k_AB, k_AC, k_BA, k_BC, k_CA, k_CB = new_k
        k_AA = 1
        k_BB = 1
        k_CC = 1

        sol = self._integrate_ODE(k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        pred_F1, pred_F2, pred_X = self._convert_XF(sol)

        plt.scatter(self.exp_data.iloc[:,0], self.exp_data.iloc[:,1])
        plt.scatter(self.exp_data.iloc[:,0], self.exp_data.iloc[:,2])
        plt.plot(pred_X,pred_F1)
        plt.plot(pred_X,pred_F2)
        plt.ylim([0,1.1])
        plt.show()


    # def reconstruct_kinetics(self, k_AA, k_AB, k_BA, k_BB):

    #     sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB)

    #     for i in range(5,10):
    #         plt.plot(sol.t, sol.y[i])
    #     plt.show()
    
    # def extract_rates(self, r_1A, r_2A, r_1B, r_2B, r_1C, r_2C):
    #     k = [1/r_1A, 1/r_2A, 1/r_1B, 1/r_2B, 1/r_1C, 1/r_2C]

    #     new_k = least_squares(fun=self._objective, x0=k, bounds=(0,20))
    #     # new_k = minimize(fun=self._objective, x0=k, method='L-BFGS-B', bounds=[(0,20),(0,20),(0,20),(0,20),(0,20),(0,20)])
    #     print("Converged rates are", new_k.x)

    #     self.display_overlay(new_k.x)
    
    def test_values(self, r_1, r_2):
        k_AB = 1/r_1
        k_BA = 1/r_2
        k_AA = 1.
        k_BB = 1.

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB)
        pred_F, pred_X = self._convert_XF(sol)

        plt.plot(pred_X,pred_F)
        plt.ylim([0,1.1])
        plt.xlabel('Total Conversion')
        plt.ylabel('Fraction Conversion')
        plt.show()


class MyOptimizationProblem(ElementwiseProblem):
    def __init__(self, polymer_obj, **kwargs):

        self.polymer_obj = polymer_obj
        super().__init__(**kwargs) 

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)

        f1, f2 = self.polymer_obj.evaluate(x)
        print(f1, f2)
        out["F"] = f1, f2  # Objective 1

def main():
    # Define the problem
    p = PetRAFTKineticFitting(exp_data, 39.77, 72., 19.48)
    problem = MyOptimizationProblem(polymer_obj=p, n_var=6, n_obj=2, n_constr=0, xl=np.array([0.1,0.1,0.1,0.1,0.1,0.1]), xu=np.array([2,2,2,2,2,2]))

    # Choose the algorithm (NSGA-II)
    algorithm = NSGA2(pop_size=100)

    # initial_guess = np.array([[0.5,0.5,0.5,0.5,0.5,0.5],[1,1,1,1,1,1], [1.1,0.9,1,1,0.9,1.2]])
    # pop = Population.new("X",initial_guess)
    # Evaluator().eval(problem, pop)

    # Perform the optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', 10),
        seed=1,
        verbose=True
    )

    # # Extract and plot the Pareto front
    # pareto_front = res.F
    # print(pareto_front)
    # Scatter(title="Pareto Front").add(pareto_front).show()



main()
# p = PetRAFTKineticFitting(exp_data, 39.77, 72., 19.48)
# p.extract_rates(1,1,1,1,1,1)