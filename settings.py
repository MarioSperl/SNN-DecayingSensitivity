import numpy as np
from auxiliary import * 
import control as ct 
import keras 


class OCP:
    def __init__(self, func_param: dict):
        self.statedim = func_param["dimension"] 
        self.interval_size = 1
        self.seed = func_param["seed"]
        self.ocp_problem_type = func_param["ocp_problem_type"]
        self.continuous = func_param["continuous"]

        # -------- matrices for optimal control problem -----------
        np.random.seed(self.seed)

        self.controldim = func_param["dimension"]
        self.bandwidth = func_param["bandwidth"]

        self.A = generate_k_banded_matrix(
            self.statedim, bandwidth=self.bandwidth)
        self.B = np.eye(self.controldim)
        self.R = np.eye(self.statedim)
        self.Q = np.eye(self.controldim) 

        if self.continuous:
            self.P, _, self.L = ct.care(self.A, self.B, self.Q, self.R)
        else:
            self.P, _, self.L = ct.dare(self.A, self.B, self.Q, self.R)
