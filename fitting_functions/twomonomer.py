import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gaussian_kde

k_s = 50
k_j = 50
k_c = 50
k_d = 0.1

class PetRAFTKineticFitting():
    def __init__(self, exp_data, A_mol, B_mol, data_index = None):
        self.exp_data = exp_data
        self.A_mol = A_mol
        self.B_mol = B_mol

        if data_index == None:
            self.data_index = 1
        else:
            self.data_index = data_index
    
    def _ODE(self, t, x, k_s, k_j, k_AA, k_AB, k_BA, k_BB, k_c, k_d):
        cta, cta_r, R_r, a, b, x_a, x_b, x_ac, x_bc, d = x
        pc = 0.01

        dxdt = np.zeros((10))

        dxdt[0] = -k_s*cta
        dxdt[1] = -k_c*cta_r*x_a - k_c*cta_r*x_b + k_s*cta
        dxdt[2] = -k_j*R_r*a - k_j*R_r*b - k_d*R_r*x_a - k_d*R_r*x_b + k_s*cta
        dxdt[3] = -k_j*R_r*a - k_AA*x_a*a - k_BA*x_b*a
        dxdt[4] = -k_j*R_r*b - k_BB*x_b*b - k_AB*x_a*b
        dxdt[5] = k_j*R_r*a + k_BA*x_b*a - k_AB*x_a*b - k_c*x_a*x_bc + k_c*x_ac*x_b - k_d*R_r*x_a - k_c*cta_r*x_a + k_c*x_ac*pc
        dxdt[6] = k_j*R_r*b + k_AB*x_a*b - k_BA*x_b*a - k_d*R_r*x_b - k_c*cta_r*x_b + k_c*x_a*x_bc - k_c*x_ac*x_b + k_c*x_bc*pc
        dxdt[7] = k_c*cta_r*x_a + k_c*x_bc*x_a - k_c*x_ac*x_b - k_c*x_ac*pc
        dxdt[8] = k_c*cta_r*x_b + k_c*x_ac*x_b - k_c*x_bc*x_a - k_c*x_bc*pc
        dxdt[9] = k_d*R_r*x_a + k_d*R_r*x_b

        return dxdt
    
    def _integrate_ODE(self, k_AA, k_AB, k_BA, k_BB, t_max):
        # initial condition
        x0 = np.zeros((10))
        x0[0] = 1.
        x0[3] = self.A_mol
        x0[4] = self.B_mol

        # parameters
        param_tuple = (k_s, k_j, k_AA, k_AB, k_BA, k_BB, k_c, k_d)
        t_span = (0, t_max)
        t_eval = np.linspace(0, t_max, int(t_max*10))

        sol = solve_ivp(self._ODE, t_span, x0, args=param_tuple, t_eval=t_eval)

        return sol
    
    def _estimate_density(self, x):
        kde = gaussian_kde(x)  # Estimate density
        density = kde(x)  # Compute density at each point
        weights = 1 / (density + 1e-6)  # Avoid division by zero
        return weights / weights.sum()  # Normalize
    
    
    def _convert_XF(self, sol):
        A_conc = sol.y[3]
        B_conc = sol.y[4]

        conv_A = A_conc / self.A_mol
        conv_B = B_conc / self.B_mol

        totalconv = 1 - ((A_conc + B_conc) / (self.A_mol + self.B_mol))

        indices = np.where(totalconv > self.exp_data.iloc[-1,0])[0]  # Get indices where condition is met
        idx = indices[0] if indices.size > 0 else -1  # Return first valid index or -1 if none found

        if idx == -1 or idx + 1 == totalconv.shape:
            return np.array([]), np.array([]), np.array([])
        else:
            idx += 1
            return conv_A[:idx], conv_B[:idx], totalconv[:idx]
    
    def _test_kinetics_convert(self, sol):
        A_conc = sol.y[3]
        B_conc = sol.y[4]

        conv_A = A_conc / self.A_mol
        conv_B = B_conc / self.B_mol

        totalconv = 1 - ((A_conc + B_conc) / (self.A_mol + self.B_mol))

        return conv_A, conv_B, totalconv
    
    def _loss(self, pred_X, pred_F, i):
        weights = self._estimate_density(self.exp_data.iloc[:,0])

        interpolator = interp1d(pred_X, pred_F, kind='linear')
        y_interpolated = interpolator(self.exp_data.iloc[:,0])
        
        grad_exp = np.diff(self.exp_data.iloc[:,i])
        grad_sim = np.diff(y_interpolated)

        ssr = np.sum(weights * (self.exp_data.iloc[:,i] - y_interpolated) ** 2)
        grad_diff = np.sum(weights[:-1] * (grad_exp - grad_sim) ** 2)
        lambda_ = 0.1

        return ssr + (lambda_ * grad_diff)
    
    def _objective(self, k):
        k_AA, k_BB = k
        k_AB = 1.
        k_BA = 1.
        t_max = 20.

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB, t_max)
        convA, convB, tot_conv = self._convert_XF(sol)

        while tot_conv.shape[0] < 20:
            t_max += 20
            sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB, t_max)
            convA, convB, tot_conv = self._convert_XF(sol)
        
        loss1 = self._loss(tot_conv, convA, 1)
        loss2 = self._loss(tot_conv, convB, 2)
        print(k)

        return loss1 + loss2
    
    def display_overlay(self, new_k, t_max = 20.):
        k_AA, k_BB = new_k
        k_AB = 1
        k_BA = 1

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB, t_max)
        convA, convB, tot_conv = self._convert_XF(sol)

        return convA, convB, tot_conv


    def reconstruct_kinetics(self, k_AA, k_BB, t_max = 20.):

        k_AB = 1
        k_BA = 1

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB, t_max)

        for i in range(3,5):
            plt.plot(sol.t, sol.y[i])
        plt.show()
    
    def predict_conversion(self, r_A, r_B, t_max = 20.):
        k_AB = 1
        k_BA = 1
        k_AA = r_A
        k_BB = r_B

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB, t_max)

        A_conv = 1 - (sol.y[3][-1] / self.A_mol)
        B_conv = 1 - (sol.y[4][-1] / self.B_mol)

        return A_conv, B_conv

    
    def extract_rates(self, r_1, r_2, t_max = 20.):
        k = [r_1, r_2]
        new_k = minimize(fun=self._objective, x0=k, method='L-BFGS-B', bounds=[(0.01,10),(0.01,10)])
        print("Converged rates are", new_k.x)

        convA, convB, tot_conv = self.display_overlay(new_k.x, t_max)

        return new_k.x, convA, convB, tot_conv
    
    def test_values(self, r_1, r_2, t_max):
        k_AB = 1.
        k_BA = 1.
        k_AA = r_1
        k_BB = r_2

        sol = self._integrate_ODE(k_AA, k_AB, k_BA, k_BB, t_max)
        convA, convB, tot_conv = self._test_kinetics_convert(sol)

        return convA, convB, tot_conv



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
    
    def predict_conversion(self, r_A, r_B):
        k_AB = 1
        k_BA = 1
        k_AA = r_A
        k_BB = r_B


    
    def extract_rates(self, k_AA, k_AB, k_BA, k_BB):
        k = [k_AA, k_AB, k_BA, k_BB]
        new_k = minimize(fun=self._objective, x0=k, method='L-BFGS-B', bounds=[(0,20),(0,20)])
        print("Converged rates are", new_k.x)

        self.display_overlay(new_k.x)

        return new_k.x