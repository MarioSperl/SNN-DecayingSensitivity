"""
A class that stores a collection of custom optimal value functions used often in
the project. This was create to allow automatic specification of the custom
function and make it possible to store and load the custom function from an
information contained in the string of the project's config file.
"""
import numpy as np
from opt_val_func import OptimalValueFunction, OptimalValueFunctionOCP
from settings import OCP

class OptimalValueFunctionCollection():
    def __init__(self):
        self.V = dict()

        # Build up and store custom functions

        # Sine-sine-rho-polynomial
        def sine_sine_rho_poly(rho):
            def sine_sine_rho_poly_wrapped(x):
                v = np.sin(x)
                I, J = np.meshgrid(v, v, indexing='ij')
                w = np.arange(len(x))
                i, j = np.meshgrid(w, w, indexing='ij')
                f = I * J * ((np.abs(i - j).astype(float) + 1) ** rho)
                return np.sum(f)
            return sine_sine_rho_poly_wrapped
        self.V["sine_sine_rho_poly"] = sine_sine_rho_poly

        # Analytical gradient vectors of sine-sine-rho polynomial
        def sine_sine_rho_poly_grad(rho):
            def sine_sine_rho_poly_grad_wrapped(X):
                grad = np.zeros(X.shape)
                for k in range(X.shape[0]):
                    x = X[k]
                    for i in range(X.shape[1]):
                        grad[k,i] = 2*np.cos(x[i])* \
                            sum([np.sin(x[j])* ((np.abs(i - j)
                                                 .astype(float) + 1) ** rho) \
                                for j in range(X.shape[1])])
                return grad
            return sine_sine_rho_poly_grad_wrapped
        self.V["sine_sine_rho_poly_grad"] = sine_sine_rho_poly_grad


        # Sine-sine-rho
        def sine_sine_rho(rho):
            def sine_sine_rho_wrapped(x):
                v = np.sin(x)
                I, J = np.meshgrid(v, v, indexing='ij')
                w = np.arange(len(x))
                i, j = np.meshgrid(w, w, indexing='ij')
                f = I * J * rho**abs(i-j)
                return np.sum(f)
            return sine_sine_rho_wrapped
        self.V["sine_sine_rho"] = sine_sine_rho

        # Analytical gradient vectors of sine-sine-rho
        def sine_sine_rho_grad(rho):
            def sine_sine_rho_grad_wrapped(X):
                grad = np.zeros(X.shape)
                for k in range(X.shape[0]):
                    x = X[k]
                    for i in range(X.shape[1]):
                        grad[k,i] = 2*np.cos(x[i])* \
                            sum([np.sin(x[j])*rho**abs(i-j) \
                                for j in range(X.shape[1])])
                return grad
            return sine_sine_rho_grad_wrapped
        self.V["sine_sine_rho_grad"] = sine_sine_rho_grad

        # State-dependent LQR
        import scipy.sparse as sp
        import scipy.linalg as la
        def state_dependent_lqr(a, b, sigma, gamma, coeff_nl, Q_scale):
            def state_dependent_lqr_wrapped(x_input: np.ndarray):
                # Preliminaries
                N = len(x_input)
                x = np.linspace(a, b, N)
                dx = x[1]-x[0]
                e = np.ones(N)

                # Generate matrices B, Q, and R
                B = np.eye(N)
                Q = Q_scale * dx * np.eye(N)
                R = gamma * Q

                # Construct sparse tridiagonal matrix A_const 
                # Dynamics are \dot x = A_const x + coeff_nl * 'nonlinearity'
                diagonals = [e, -2*e, e]
                A_const = sp.diags(diagonals,
                                   offsets=[-1, 0, 1], shape=(N, N)).toarray()
                A_const[0, 0] = -1
                A_const[-1, -1] = -1
                A_const = sigma * A_const / dx**2

                # Ax operator: A_const plus nonlinear state-dependent diagonal
                def Ax(x_vec):
                    return A_const + coeff_nl * np.diag(1 - x_vec**2)
                
                # Define function to compute solution of Riccati equation and
                # evaluate value function
                def P(x_vec):
                    A_mat = Ax(x_vec)
                    return la.solve_continuous_are(A_mat, B, Q, R)

                def V(x_vec):
                    P_mat = P(x_vec)
                    return x_vec.T @ P_mat @ x_vec

                return V(x_input)
            return state_dependent_lqr_wrapped
        self.V["state_dependent_lqr"] = state_dependent_lqr

        # State-dependent LQR gradient
        from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov
        def state_dependent_lqr_grad(a, b, sigma, gamma, coeff_nl, Q_scale):
            def state_dependent_lqr_grad_wrapped(X: np.ndarray):
                grad = np.zeros_like(X)
                N = X.shape[1]
                dx = (b - a) / (N - 1)
                e = np.ones(N)
                Q = Q_scale * dx * np.eye(N)
                R = gamma * Q
                invR = np.linalg.inv(R)
                BRinvB = np.eye(N) @ invR @ np.eye(N).T

                # constant part of A
                A_const = sp.diags([e, -2*e, e], [-1, 0, 1],
                                   shape=(N, N)).toarray()
                A_const[0, 0] = -1
                A_const[-1, -1] = -1
                A_const = sigma * A_const / dx**2

                for k in range(X.shape[0]):
                    x_vec = X[k]
                    Ax = A_const + coeff_nl * np.diag(1 - x_vec**2)
                    P_mat = solve_continuous_are(Ax, np.eye(N), Q, R)
                    Acl = Ax - BRinvB @ P_mat
                    grad_k = 2 * (P_mat @ x_vec)

                    for i in range(N):
                        dAi = np.zeros((N, N))
                        dAi[i, i] = -2 * coeff_nl * x_vec[i]

                        RHS = -(dAi.T @ P_mat + P_mat @ dAi)
                        Mi = solve_continuous_lyapunov(Acl.T, RHS)
                        grad_k[i] += x_vec @ Mi @ x_vec

                    grad[k] = grad_k

                return grad
            return state_dependent_lqr_grad_wrapped
        self.V["state_dependent_lqr_grad"] = state_dependent_lqr_grad
    
    def get_func(self, func_param: dict):
        if "function" in func_param.keys():
            key = func_param["function"]
            if key in self.V.keys():
                if key == "sine_sine_rho":
                    if func_param["gradient_provided"]:
                        return OptimalValueFunction(
                            self.V["sine_sine_rho"](func_param["rho"]),
                            gradient_function=self.V["sine_sine_rho_grad"](
                                func_param["rho"])
                        )
                    else:
                        return OptimalValueFunction(
                            self.V["sine_sine_rho"](func_param["rho"]))

                elif key == "sine_sine_rho_poly":
                    return OptimalValueFunction(
                            self.V["sine_sine_rho_poly"](func_param["rho"]),
                            gradient_function=self.V["sine_sine_rho_poly_grad"](
                                func_param["rho"])
                        )

                elif key == "state_dependent_lqr":
                    if func_param["gradient_provided"]:
                        grad = self.V["state_dependent_lqr_grad"](
                                        a=func_param["a"],
                                        b=func_param["b"],
                                        sigma=func_param["sigma"],
                                        gamma=func_param["gamma"],
                                        coeff_nl=func_param["coeff_nl"],
                                        Q_scale=func_param["Q_scale"],
                                    )
                    else:
                        grad = None

                    return OptimalValueFunction(
                            self.V["state_dependent_lqr"](
                                    a=func_param["a"],
                                    b=func_param["b"],
                                    sigma=func_param["sigma"],
                                    gamma=func_param["gamma"],
                                    coeff_nl=func_param["coeff_nl"],
                                    Q_scale=func_param["Q_scale"],
                            ), gradient_function=grad)
        
            elif key == "linear-quadratic-ocp":
                ocp = OCP(func_param)
                return OptimalValueFunctionOCP(ocp)
            else:
                raise KeyError(f"Optimal value function '{key}' not found!")
        else:
            raise KeyError(f"Optimal value function not specified!")
