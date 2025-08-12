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

        # General OCP problem
        if self.ocp_problem_type == "general":
            self.controldim = func_param["dimension"]
            self.bandwidth = func_param["bandwidth"]

            self.A = generate_k_banded_matrix(
                self.statedim, bandwidth=self.bandwidth)
            self.B = np.eye(self.controldim)
            self.R = np.eye(self.statedim)
            self.Q = np.eye(self.controldim) 

        # Vehicle LQR problem
        elif self.ocp_problem_type == "vehicle":
            self.num_vehicles = func_param["num_vehicles"] 
            self.inputdim = 2 * self.num_vehicles
            self.controldim = self.num_vehicles

            # Prevent mismatch between parameter dimension and num_vehicles
            if func_param["dimension"] != 2*func_param["num_vehicles"]:
                raise ValueError("Dimension must be 2 * num_vehicles")

            # Generation of A, B, Q, and R matrices from Mario's script
            self.B = np.zeros((self.inputdim, self.controldim))
            for i in range(self.controldim): 
                self.B[2*i+1, i] = 1 

            A_block = np.array([[0, 1], [0, 0]])
            self.A = np.kron(np.eye(self.num_vehicles), A_block)

            self.R = 0.1*np.eye(self.controldim)

            Q_block_diag = np.array([[2, 0], [0, 1]])
            Q_block_offdiag = np.array([[-1, 0], [0, 0]])

            # Compute the Kronecker products
            kron_diag = np.kron(np.eye(self.num_vehicles), Q_block_diag)
            kron_offdiag1 = np.kron(np.diag(
                np.ones(self.num_vehicles-1), 1), Q_block_offdiag)
            kron_offdiag2 = np.kron(np.diag(
                np.ones(self.num_vehicles-1), -1), Q_block_offdiag)
            
            self.Q = kron_diag + kron_offdiag1 + kron_offdiag2

            self.Q[2*self.num_vehicles-2, 2*self.num_vehicles-2] = 1
            self.Q[0,0] = 1 
            Q_block_all_references = np.array([[1, 0], [0, 0]])
            self.Q = self.Q + np.kron(
                np.eye(self.num_vehicles), Q_block_all_references)
        else:
            raise NotImplementedError("Problem type not implemented!")

        if self.continuous:
            self.P, _, self.L = ct.care(self.A, self.B, self.Q, self.R)
        else:
            self.P, _, self.L = ct.dare(self.A, self.B, self.Q, self.R)
