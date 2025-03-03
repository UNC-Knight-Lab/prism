import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.optimize import least_squares, minimize

k_s = 100
k_j = 100
k_c = 100
k_d = 0.1

class ThreeMonomerThermalRAFTKineticFitting():
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
        t_span = (0, 100.0)
        t_eval = np.linspace(0, 100., 100)

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
        fracC = f_C / (f_A + f_B + f_C)
        totalfrac = ((f_iA - f_A) + (f_iB - f_B) + (f_iC - f_C)) / (f_iA + f_iB + f_iC)

        idx = np.argmax(totalfrac > 0.96)

        return fracA[:idx], fracB[:idx], totalfrac[:idx]
    
    def _sum_square_residuals(self, pred_X, pred_F, i):
        interpolator = interp1d(pred_X, pred_F, kind='linear')
        y_interpolated = interpolator(self.exp_data.iloc[:,0])

        residuals = self.exp_data.iloc[:,i] - y_interpolated

        return np.sum(residuals**2)
    
    def _objective1(self, k):
        k_AB, k_AC, k_BA, k_BC, k_CA, k_CB = k
        k_AA = 1.
        k_BB = 1.
        k_CC = 1.

        print(k)
        sol = self._integrate_ODE(k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        pred_F1, pred_F2, pred_X = self._convert_XF(sol)
        loss2 = self._sum_square_residuals(pred_X, pred_F2, 2)

        loss1 = self._sum_square_residuals(pred_X, pred_F1, 1)

        return loss1 + loss2
    
    def _objective2(self, k):
        k_AB, k_AC, k_BA, k_BC, k_CA, k_CB = k
        k_AA = 1.
        k_BB = 1.
        k_CC = 1.

        sol = self._integrate_ODE(k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        pred_F1, pred_F2, pred_X = self._convert_XF(sol)

        loss2 = self._sum_square_residuals(pred_X, pred_F2, 2)

        return loss2
    
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

    
    def extract_rates(self, r_1A, r_2A, r_1B, r_2B, r_1C, r_2C):
        k = [1/r_1A, 1/r_2A, 1/r_1B, 1/r_2B, 1/r_1C, 1/r_2C]

        k = minimize(fun=self._objective1, x0=k, method='BFGS', bounds=[(0,20),(0,20),(0,20),(0,20),(0,20),(0,20)], options={'maxiter': 10})
        # k = minimize(fun=self._objective2, x0=k.x, method='CG', bounds=[(0,20),(0,20),(0,20),(0,20),(0,20),(0,20)], options={'maxiter': 1})
        # k = minimize(fun=self._objective1, x0=k.x, method='CG', bounds=[(0,20),(0,20),(0,20),(0,20),(0,20),(0,20)], options={'maxiter': 1})
        print("Converged rates are", k.x)

        self.display_overlay(k.x)
    
    def test_values(self, r_1A, r_2A, r_1B, r_2B, r_1C, r_2C):
        k_AB = 1/r_1A
        k_AC = 1/r_2A
        k_BA = 1/r_1B
        k_BC = 1/r_2B
        k_CA = 1/r_1C
        k_CB = 1/r_2C
        k_AA = 1.
        k_BB = 1.
        k_CC = 1.

        sol = self._integrate_ODE(k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        pred_F1, pred_F2, pred_X = self._convert_XF(sol)

        plt.plot(pred_X,pred_F1)
        plt.plot(pred_X,pred_F2)
        plt.ylim([0,1.1])
        plt.show()

class ThreeMonomerPETRAFTKineticFitting():
    def __init__(self, exp_data, A_mol, B_mol, C_mol):
        self.exp_data = exp_data
        self.A_mol = A_mol
        self.B_mol = B_mol
        self.C_mol = C_mol
    
    def _ODE(self, t, x, k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d):
        cta, cta_r, R_r, a, b, c, x_a, x_b, x_c, x_ac, x_bc, x_cc, d = x

        dxdt = np.zeros((13))

        dxdt[0] = -k_s*cta
        dxdt[1] = -k_c*cta_r*(x_a + x_b + x_c) + k_s*cta
        dxdt[2] = -k_j*R_r*(a + b + c) - k_d*R_r*(x_a + x_b + x_c) + k_s*cta
        dxdt[3] = -k_j*R_r*a - k_AA*x_a*a - k_BA*x_b*a - k_CA*x_c*a
        dxdt[4] = -k_j*R_r*b - k_AB*x_a*b - k_BB*x_b*b - k_CB*x_c*b
        dxdt[5] = -k_j*R_r*c - k_AC*x_a*c - k_BC*x_b*c - k_CC*x_c*c
        dxdt[6] = k_j*R_r*a + k_BA*x_b*a - k_AB*x_a*b - k_AC*x_a*c + k_CA*x_c*a - k_c*x_a*cta - k_c*x_a*x_bc + k_c*x_ac*x_b + k_c*x_ac*x_c - k_c*x_cc*x_a - k_d*R_r*x_a + k_c*x_ac
        dxdt[7] = k_j*R_r*b - k_BA*x_b*a + k_AB*x_a*b - k_BC*x_b*c + k_CB*x_c*b - k_c*x_b*cta + k_c*x_a*x_bc - k_c*x_ac*x_b + k_c*x_bc*x_c - k_c*x_cc*x_b - k_d*R_r*x_b + k_c*x_bc
        dxdt[8] = k_j*R_r*c + k_AC*x_a*c + k_BC*x_b*c - k_CA*x_c*a - k_CB*x_c*b - k_c*x_c*cta - k_c*x_bc*x_c + k_c*x_cc*x_b - k_c*x_ac*x_c + k_c*x_cc*x_a - k_d*R_r*x_c + k_c*x_cc
        dxdt[9] = k_c*x_a*cta - k_c*x_ac*x_b + k_c*x_bc*x_a - k_c*x_ac*c + k_c*x_cc*x_a - k_c*x_ac
        dxdt[10] = k_c*x_b*cta - k_c*x_bc*x_c + k_c*x_cc*x_b + k_c*x_ac*x_b - k_c*x_bc*x_a - k_c*x_bc
        dxdt[11] = k_c*x_c*cta + k_c*x_bc*x_c - k_c*x_cc*x_b + k_c*x_ac*x_c - k_c*x_cc*x_a - k_c*x_cc
        dxdt[12] = k_d*R_r*(x_a + x_b + x_c)

        return dxdt
    
    def _integrate_ODE(self, k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d):
        # initial condition
        x0 = np.zeros((13))
        x0[0] = 1.
        x0[3] = self.A_mol
        x0[4] = self.B_mol
        x0[5] = self.C_mol

        # parameters
        param_tuple = (k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        t_span = (0, 100.0)
        t_eval = np.linspace(0, 100., 100)

        sol = solve_ivp(self._ODE, t_span, x0, args=param_tuple, t_eval=t_eval)

        return sol
    
    def _convert_XF(self, sol):
        f_iA = self.A_mol / (self.A_mol + self.B_mol + self.C_mol)
        f_iB = self.B_mol / (self.A_mol + self.B_mol + self.C_mol)
        f_iC = self.C_mol / (self.A_mol + self.B_mol + self.C_mol)
    
        A_conc = sol.y[3]
        B_conc = sol.y[4]
        C_conc = sol.y[5]

        f_A = (A_conc / self.A_mol) * f_iA
        f_B = (B_conc / self.B_mol) * f_iB
        f_C = (C_conc / self.C_mol) * f_iC

        f_A[0] = f_iA
        f_B[0] = f_iB
        f_C[0] = f_iC

        fracA = f_A / (f_A + f_B + f_C)
        fracB = f_B / (f_A + f_B + f_C)
        fracC = f_C / (f_A + f_B + f_C)
        totalfrac = ((f_iA - f_A) + (f_iB - f_B) + (f_iC - f_C)) / (f_iA + f_iB + f_iC)

        idx = np.argmax(totalfrac > 0.96)

        return fracA[:idx], fracB[:idx], totalfrac[:idx]
    
    def _sum_square_residuals(self, pred_X, pred_F, i):
        interpolator = interp1d(pred_X, pred_F, kind='linear')
        y_interpolated = interpolator(self.exp_data.iloc[:,0])

        residuals = self.exp_data.iloc[:,i] - y_interpolated

        return np.sum(residuals**2)
    
    def _objective1(self, k):
        k_AB, k_AC, k_BA, k_BC, k_CA, k_CB = k
        k_AA = 1.
        k_BB = 1.
        k_CC = 1.

        print(k)
        sol = self._integrate_ODE(k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        pred_F1, pred_F2, pred_X = self._convert_XF(sol)
        loss2 = self._sum_square_residuals(pred_X, pred_F2, 2)

        loss1 = self._sum_square_residuals(pred_X, pred_F1, 1)

        return loss1 + loss2
    
    def _objective2(self, k):
        k_AB, k_AC, k_BA, k_BC, k_CA, k_CB = k
        k_AA = 1.
        k_BB = 1.
        k_CC = 1.

        sol = self._integrate_ODE(k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        pred_F1, pred_F2, pred_X = self._convert_XF(sol)

        loss2 = self._sum_square_residuals(pred_X, pred_F2, 2)

        return loss2
    
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

    
    def extract_rates(self, r_1A, r_2A, r_1B, r_2B, r_1C, r_2C):
        k = [1/r_1A, 1/r_2A, 1/r_1B, 1/r_2B, 1/r_1C, 1/r_2C]

        k = minimize(fun=self._objective1, x0=k, method='BFGS', bounds=[(0,20),(0,20),(0,20),(0,20),(0,20),(0,20)], options={'maxiter': 10})
        # k = minimize(fun=self._objective2, x0=k.x, method='CG', bounds=[(0,20),(0,20),(0,20),(0,20),(0,20),(0,20)], options={'maxiter': 1})
        # k = minimize(fun=self._objective1, x0=k.x, method='CG', bounds=[(0,20),(0,20),(0,20),(0,20),(0,20),(0,20)], options={'maxiter': 1})
        print("Converged rates are", k.x)

        self.display_overlay(k.x)
    
    def test_values(self, r_1A, r_2A, r_1B, r_2B, r_1C, r_2C):
        k_AB = 1/r_1A
        k_AC = 1/r_2A
        k_BA = 1/r_1B
        k_BC = 1/r_2B
        k_CA = 1/r_1C
        k_CB = 1/r_2C
        k_AA = 1.
        k_BB = 1.
        k_CC = 1.

        sol = self._integrate_ODE(k_s, k_j, k_AA, k_AB, k_AC, k_BB, k_BA, k_BC, k_CC, k_CA, k_CB, k_c, k_d)
        print(sol.y[3])
        pred_F1, pred_F2, pred_X = self._convert_XF(sol)

        plt.plot(pred_X,pred_F1)
        plt.plot(pred_X,pred_F2)
        plt.ylim([0,1.1])
        plt.show()