import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time

class ASM1():

    def __init__(self, volume_tank = 1000):

        # Set initial conditions
        self.X_names = ["S_I", "S_S", "X_I", "X_S", "X_{B,H}", "X_{B,A}", "X_P", "S_O", "S_{NO}", "S_{NH}", "S_{ND}", "X_{ND}", "S_{ALK}"]
        self.set_initial_conditions()

        # Initialize the dictionaries of parameters
        self.set_parameters()
        self.vol = volume_tank

    def set_initial_conditions(self):

        # Data collected from the simulations from BSM1 => to validate
        self.X_init = np.array([28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213])

    def set_kinetic_parameters(self, params=None):

        # Data collected from the simulations from BSM1 => to validate
        self.fifteen_degrees_kinetic_parameters = {"\mu_{H}" : 4.0,
                                    "K_S" : 10.0,
                                    "K_{OH}" : 0.2,
                                    "K_{NO}" : 0.5,
                                    "b_H" : 0.3,
                                    "\eta_{g}" : 0.8,
                                    "\eta_{h}" : 0.8,
                                    "k_h" : 3.0,
                                    "K_X" : 0.1,
                                    "\mu_{A}" : 0.5,
                                    "K_{NH}" : 1.0,
                                    "b_A" : 0.05,
                                    "K_{OA}" :  0.4,
                                    "k_{a}" : 0.05}

        self.temperature_coefficients = {"\mu_{H}" : 3.0,
                                    "K_S" : 0,
                                    "K_{OH}" : 0,
                                    "K_{NO}" : 0,
                                    "b_H" : 0.2,
                                    "\eta_{g}" : 0,
                                    "\eta_{h}" : 0,
                                    "k_h" : 2.5,
                                    "K_X" : 0,
                                    "\mu_{A}" : 0.3,
                                    "K_{NH}" : 0,
                                    "b_A" : 0.03,
                                    "K_{OA}" :  0,
                                    "k_{a}" : 0.04}

        if params is None:
            self.kinetic_parameters = self.fifteen_degrees_kinetic_parameters

    def set_stoichiometric_parameters(self, params=None):

        if params is None:
            # Data collected from the simulations from BSM1 => to validate
            self.stoichiometric_parameters = {"Y_A" : 0.24,
                                              "Y_H" : 0.67,
                                              "f_P" : 0.08,
                                              "i_{XB}" : 0.08,
                                              "i_{XP}" : 0.06}

        # Gather stoichiometric parameters
        Y_A = self.stoichiometric_parameters["Y_A"]
        Y_H = self.stoichiometric_parameters["Y_H"]
        f_P = self.stoichiometric_parameters["f_P"]
        i_XB = self.stoichiometric_parameters["i_{XB}"]
        i_XP = self.stoichiometric_parameters["i_{XP}"]

        # Set stoichiometric matrix
        self.R = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
            [-1/Y_H, -1/Y_H, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1-f_P, 1- f_P, 0, -1, 0],
            [1, 1, 0, -1, 0, 0, 0, 0],
            [0, 0, 1, 0, -1, 0, 0, 0],
            [0, 0, 0, f_P, f_P, 0, 0, 0],
            [-((1-Y_H)/Y_H), 0, (-(4.57/Y_A)+1.0), 0, 0, 0, 0, 0],
            [0, -((1-Y_H)/(2.86*Y_H)), (1.0/Y_A), 0, 0, 0, 0, 0],
            [-i_XB, -i_XB, -(i_XB+(1.0/Y_A)), 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 1],
            [0, 0, 0, (i_XB-f_P*i_XP), (i_XB-f_P*i_XP), 0, 0, -1],
            [-i_XB/14.0, ((1.0-Y_H)/(14.0*2.86*Y_H)-(i_XB/14.0)), -((i_XB/14.0)+1.0/(7.0*Y_A)), 0, 0, 1/14, 0, 0]])

    def set_parameters(self, params=None):

        self.set_kinetic_parameters(params)
        self.set_stoichiometric_parameters(params)

    def alter_parameters(self, temperature):

        for idx in self.kinetic_parameters.keys():
            if self.temperature_coefficients[idx] != 0:
                self.kinetic_parameters[idx] = self.fifteen_degrees_kinetic_parameters[idx]*np.exp(np.log(self.fifteen_degrees_kinetic_parameters[idx]/self.temperature_coefficients[idx])*(temperature-15))

    def _compute_process_rates(self, t, X):

        # Gather kinetic parameters
        mu_H = self.kinetic_parameters["\mu_{H}"]
        K_S = self.kinetic_parameters["K_S"]
        K_OH = self.kinetic_parameters["K_{OH}"]
        K_NO = self.kinetic_parameters["K_{NO}"]
        b_H = self.kinetic_parameters["b_H"]
        eta_g = self.kinetic_parameters["\eta_{g}"]
        eta_h = self.kinetic_parameters["\eta_{h}"]
        k_h = self.kinetic_parameters["k_h"]
        K_X = self.kinetic_parameters["K_X"]
        mu_A = self.kinetic_parameters["\mu_{A}"]
        K_NH = self.kinetic_parameters["K_{NH}"]
        b_A = self.kinetic_parameters["b_A"]
        K_OA = self.kinetic_parameters["K_{OA}"]
        k_a = self.kinetic_parameters["k_{a}"]

        # Compute process rates
        process_rates= np.array([mu_H*(X[1]/(K_S+X[1]))*(X[7]/(K_OH+X[7]))*X[4], # Aerobic growth of heterotrophs
                            mu_H*(X[1]/(K_S+X[1]))*(K_OH/(K_OH+X[7]))*(X[8]/(K_NO+X[8]))*eta_g*X[4], # Anoxic growth of heterotrophs
                            mu_A*(X[9]/(K_NH+X[9]))*(X[7]/(K_OA+X[7]))*X[5], # Aerobic growth of autotrophs
                            b_H*X[4], # "Decay" of heterotrophs
                            b_A*X[5], # "Decay" of autotrophs
                            k_a*X[10]*X[4], # Ammonification of soluble organic nitrogen
                            k_h*((X[3]/X[4])/(K_X+(X[3]/X[4])))*((X[7]/(K_OH+X[7]))+eta_h*(K_OH/(K_OH+X[7]))*(X[8]/(K_NO+X[8])))*X[4], # "Hydrolysis" of entrapped organics
                            (k_h*((X[3]/X[4])/(K_X+(X[3]/X[4])))*((X[7]/(K_OH+X[7]))+eta_h*(K_OH/(K_OH+X[7]))*(X[8]/(K_NO+X[8])))*X[4])*X[11]/X[3]]) # "Hydrolysis" of entrapped organics nitrogen

        return process_rates

    def _compute_reaction_rates(self, process_rates, X):

        
        # reaction_rates = np.array([0,
        #                     (-process_rates[0]-process_rates[1])/Y_H + process_rates[6],
        #                     0,
        #                     (1.0-f_P)*(process_rates[3] + process_rates[4]) - process_rates[6],
        #                     process_rates[0] + process_rates[1] - process_rates[3],
        #                     process_rates[2] - process_rates[4],
        #                     f_P*(process_rates[3] + process_rates[4]),
        #                     -((1-Y_H)/Y_H)*process_rates[0] + (-(4.57/Y_A)+1.0)*process_rates[2], 
        #                     -((1-Y_H)/(2.86*Y_H))*process_rates[1] + (1.0/Y_A)*process_rates[2],
        #                     -i_XB*(process_rates[0] + process_rates[1]) -(i_XB+(1.0/Y_A))*process_rates[2] + process_rates[5],
        #                     -process_rates[5] + process_rates[7],
        #                     (i_XB-f_P*i_XP)*(process_rates[3] + process_rates[4]) - process_rates[7],
        #                     -i_XB/14.0*process_rates[0]+((1.0-Y_H)/(14.0*2.86*Y_H)-(i_XB/14.0))*process_rates[1]-((i_XB/14.0)+1.0/(7.0*Y_A))*process_rates[2]+process_rates[5]/14.0])
        # Compute reaction rates
        return self.R.dot(process_rates) 

    def _compute_derivatives(self, t, X, X_input, Q_in, oxygen_input_function=None):

        # Compute derivatives
        X_derivatives = (Q_in/self.vol)*(X_input - X) + self._compute_reaction_rates(self._compute_process_rates(t, X), X) 

        # Get oxygen input
        if oxygen_input_function is not None:
            X_derivatives[7] = X_derivatives[7] + oxygen_input_function(t, X)

        # Saturation of oxygen
        X_derivatives[7] = X_derivatives[7] - 240 * max(X[7] - 8, 0)

        return X_derivatives

    def simulate(self, tspan, X_input, Q_in, oxygen_input_function=None, **kwargs):

        # Solve ODEs
        sol = solve_ivp(self._compute_derivatives, tspan, self.X_init, args=(X_input, Q_in, oxygen_input_function), **kwargs)

        return sol


def oxygen_input_function(t, X):
    """
    Control oxygen strategy
    
    """

    if t%1 < 0.3 or ( t%1 > 0.4 and t%1 < 0.5) or ( t%1 > 0.8 and t%1 < 0.9):
        return 350
    else:
        return 0

if __name__ == "__main__":

    # Create model
    model = ASM1()
    X_in = np.array([28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213])
    Q_in = 226

    # Simulate
    t1 = time.time_ns()
    sol = model.simulate([0, 20], X_in, Q_in, oxygen_input_function, atol=1e-8, rtol=1e-8, t_eval=np.linspace(10, 20, 1000))
    t2 = time.time_ns()
    print("Time: ", (t2-t1)/1e9,"s")

    # Plot results
    plt.fill_between(sol.t, sol.y[7]/2, label=model.X_names[7], alpha=0.3)
    plt.plot(sol.t, sol.y[8], label=model.X_names[8])
    plt.plot(sol.t, sol.y[9], label=model.X_names[9])
    plt.legend()
    plt.show()