import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.optimize import minimize

k_s = 100
k_j = 100
k_c = 100
k_d = 0.1

class PetRAFTKineticFitting():
    def __init__(self, exp_data, A_mol, B_mol):
        self.exp_data = exp_data
        self.A_mol = A_mol
        self.B_mol = B_mol
    
    def _ODE(self, t, x, k_s, k_j, k_AA, k_AB, k_BA, k_BB, k_c, k_d):
        cta, cta_r, R_r, a, b, x_a, x_b, x_ac, x_bc, d = x

        dxdt = np.zeros((10))

        dxdt[0] = -k_s*cta
        dxdt[1] = -k_c*cta_r*x_a - k_c*cta_r*x_b + k_s*cta
        dxdt[2] = -k_j*R_r*a - k_j*R_r*b - k_d*R_r*x_a - k_d*R_r*x_b + k_s*cta
        dxdt[3] = -k_j*R_r*a - k_AA*x_a*a - k_BA*x_b*a
        dxdt[4] = -k_j*R_r*b - k_BB*x_b*b - k_AB*x_a*b
        dxdt[5] = k_j*R_r*a + k_BA*x_b*a - k_AB*x_a*b - k_c*x_a*x_bc + k_c*x_ac*x_b - k_d*R_r*x_a - k_c*cta_r*x_a + k_c*x_ac
        dxdt[6] = k_j*R_r*b + k_AB*x_a*b - k_BA*x_b*a - k_d*R_r*x_b - k_c*cta_r*x_b + k_c*x_a*x_bc - k_c*x_ac*x_b + k_c*x_bc
        dxdt[7] = k_c*cta_r*x_a + k_c*x_bc*x_a - k_c*x_ac*x_b - k_c*x_ac
        dxdt[8] = k_c*cta_r*x_b + k_c*x_ac*x_b - k_c*x_bc*x_a - k_c*x_bc
        dxdt[9] = k_d*R_r*x_a + k_d*R_r*x_b

        return dxdt
    
    def _integrate_ODE(self, k_AA, k_AB, k_BA, k_BB):
        # initial condition
        x0 = np.zeros((10))
        x0[0] = 1.
        x0[3] = self.A_mol
        x0[4] = self.B_mol

        # parameters
        param_tuple = (k_s, k_j, k_AA, k_AB, k_BA, k_BB, k_c, k_d)
        t_span = (0, 10.0)
        t_eval = np.linspace(0, 10., 100)

        sol = solve_ivp(self._ODE, t_span, x0, args=param_tuple, t_eval=t_eval)

        return sol
    
    def _convert_XF(self, sol):
        f_iA = self.A_mol / (self.A_mol + self.B_mol)
        f_iB = self.B_mol / (self.A_mol + self.B_mol)

        A_conc = sol.y[3]
        B_conc = sol.y[4]

        f_A = (A_conc / self.A_mol) * f_iA
        f_B = (B_conc / self.B_mol) * f_iB

        f_A[0] = f_iA
        f_B[0] = f_iB

        fracA = f_A / (f_A + f_B)
        totalfrac = ((f_iA - f_A) + (f_iB - f_B)) / (f_iA + f_iB)

        idx = np.argmax(totalfrac > 0.95)

        return fracA[:idx], totalfrac[:idx]
    
    def _sum_square_residuals(self, pred_X, pred_F):
        interpolator = interp1d(pred_X, pred_F, kind='linear')
        y_interpolated = interpolator(self.exp_data.iloc[:,0])

        residuals = self.exp_data.iloc[:,1] - y_interpolated

        return np.sum(residuals**2)
    
    def _objective(self, k):
        k_AB, k_BA = k
        k_AA = 1.
        k_BB = 1.

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB)
        pred_F, pred_X = self._convert_XF(sol)
        loss = self._sum_square_residuals(pred_X, pred_F)

        return loss
    
    def display_overlay(self, new_k):
        k_AB, k_BA = new_k
        k_AA = 1
        k_BB = 1

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB)
        pred_F, pred_X = self._convert_XF(sol)

        plt.scatter(self.exp_data.iloc[:,0], self.exp_data.iloc[:,1])
        plt.plot(pred_X,pred_F)
        plt.ylim([0,1.1])
        plt.show()


    def reconstruct_kinetics(self, k_AA, k_AB, k_BA, k_BB):

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB)

        for i in range(5,10):
            plt.plot(sol.t, sol.y[i])
        plt.show()
    
    def extract_rates(self, r_1, r_2):
        k = [1/r_1, 1/r_2]
        new_k = minimize(fun=self._objective, x0=k, method='L-BFGS-B', bounds=[(0,20),(0,20)])
        print("Converged rates are", new_k.x)

        self.display_overlay(new_k.x)
    
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




class ThermalRAFTKineticFitting():
    def __init__(self, exp_data, A_mol, B_mol):
        self.exp_data = exp_data
        self.A_mol = A_mol
        self.B_mol = B_mol
    
    def _ODE(self, t, x, k_s, k_j, k_AA, k_AB, k_BA, k_BB, k_c, k_d):
        i, i_r, cta, R_r, a, b, x_a, x_b, x_ac, x_bc, d = x

        dxdt = np.zeros((11))

        dxdt[0] = -k_s*i
        dxdt[1] = k_s*i - k_j*i_r*a - k_j*i_r*b - k_d*x_a*i_r - k_d*i_r*x_b
        dxdt[2] = -k_c*x_a*cta - k_c*x_b*cta
        dxdt[3] = k_c*cta*x_a + k_c*cta*x_b - k_j*R_r*a - k_j*R_r*b - k_d*R_r*x_a - k_d*x_b*R_r
        dxdt[4] = -k_j*i_r*a - k_j*R_r*a - k_AA*x_a*a - k_BA*x_b*a
        dxdt[5] = -k_j*R_r*b - k_j*i_r*b - k_BB*x_b*b - k_AB*x_a*b
        dxdt[6] = k_j*i_r*a + k_j*R_r*a + k_BA*x_b*a - k_AB*x_a*b - k_c*x_a*x_bc + k_c*x_ac*x_b - k_c*cta*x_a - k_d*R_r*x_a - k_d*i_r*x_a
        dxdt[7] = k_j*i_r*b + k_j*R_r*b + k_AB*x_a*b - k_BA*x_b*a + k_c*x_a*x_bc - k_c*x_ac*x_b + k_c*cta*x_b - k_d*R_r*x_b - k_d*i_r*x_b

        dxdt[8] = k_c*cta*x_a + k_c*x_bc*x_a - k_c*x_ac*x_b
        dxdt[8] = k_c*cta_r*x_b + k_c*x_ac*x_b - k_c*x_bc*x_a
        dxdt[9] = k_d*R_r*x_a + k_d*R_r*x_b


        return dxdt
    
    def _integrate_ODE(self, k_AA, k_AB, k_BA, k_BB):
        # initial condition
        x0 = np.zeros((10))
        x0[0] = 1.
        x0[3] = self.A_mol
        x0[4] = self.B_mol

        # parameters
        param_tuple = (k_s, k_j, k_AA, k_AB, k_BA, k_BB, k_c, k_d)
        t_span = (0, 10.0)
        t_eval = np.linspace(0, 10., 100)

        sol = solve_ivp(self._ODE, t_span, x0, args=param_tuple, t_eval=t_eval)

        return sol
    
    def _convert_XF(self, sol):
        f_iA = self.A_mol / (self.A_mol + self.B_mol)
        f_iB = self.B_mol / (self.A_mol + self.B_mol)

        A_conc = sol.y[3]
        B_conc = sol.y[4]

        f_A = (A_conc / self.A_mol) * f_iA
        f_B = (B_conc / self.B_mol) * f_iB

        f_A[0] = f_iA
        f_B[0] = f_iB

        fracA = f_A / (f_A + f_B)
        totalfrac = ((f_iA - f_A) + (f_iB - f_B)) / (f_iA + f_iB)

        idx = np.argmax(totalfrac > 0.85)

        return fracA[:idx], totalfrac[:idx]
    
    def _sum_square_residuals(self, pred_X, pred_F):
        interpolator = interp1d(pred_X, pred_F, kind='linear')
        y_interpolated = interpolator(self.exp_data.iloc[:,0])

        residuals = self.exp_data.iloc[:,1] - y_interpolated

        return np.sum(residuals**2)
    
    def _objective(self, k):
        k_AA, k_AB, k_BA, k_BB = k

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB)
        pred_F, pred_X = self._convert_XF(sol)
        loss = self._sum_square_residuals(pred_X, pred_F)

        return loss
    
    def display_overlay(self, new_k):
        k_AA, k_AB, k_BA, k_BB = new_k

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB)
        self.reconstruct_kinetics(k_AA, k_AB, k_BA, k_BB)
        pred_F, pred_X = self._convert_XF(sol)

        plt.scatter(self.exp_data.iloc[:,0], self.exp_data.iloc[:,1])
        plt.plot(pred_X,pred_F)
        plt.ylim([0,1.1])
        plt.show()


    def reconstruct_kinetics(self, k_AA, k_AB, k_BA, k_BB):

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB)

        for i in range(5,10):
            plt.plot(sol.t, sol.y[i])
        plt.show()
    
    def extract_rates(self, k_AA, k_AB, k_BA, k_BB):
        k = [k_AA, k_AB, k_BA, k_BB]
        new_k = minimize(fun=self._objective, x0=k, method='L-BFGS-B', bounds=[(0,20),(0,20)])
        print("Converged rates are", new_k.x)

        self.display_overlay(new_k.x)