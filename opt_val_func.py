import numpy as np

"""
A class for an arbitrary optimal value function V equipped with gradient
vector and Hessian matrix evaluation. Differentiation is done numerically with
central differencing.
"""
class OptimalValueFunction():
    def __init__(self, V: callable, gradient_function=None):
        """
        Parameters
        ----------
        V: callable
            The custom optimal value function described as a map from R^N to R
        gradient_function: callable
            An analytical solution of the gradient vectors given X
        """
        self.V = V
        self.gradient_function = gradient_function

    def evaluate(self, X: np.ndarray):
        self.values = np.zeros((X.shape[0],1))
        for i in range(X.shape[0]):
            self.values[i] = self.V(X[i,:]) 
        return self.values

    def evaluate_gradient(self, X: np.ndarray):
        # Calculate gradient if not supplied
        if not self.gradient_function:
            h = 1e-3
            self.grad = np.zeros(X.shape)
            # Looping over the dimension
            for i in range(X.shape[1]):
                perturbed = np.copy(X)
                perturbed[:,i] += h
                forward = np.apply_along_axis(self.V, 1, perturbed)
                perturbed[:,i] -= 2*h
                backward = np.apply_along_axis(self.V, 1, perturbed)
                self.grad[:,i] = (forward-backward)/(2*h)
            return self.grad
        else:
            return self.gradient_function(X)

    def evaluate_hessian(self, X: np.ndarray):
        h = 1e-3
        self.hessian = np.zeros((X.shape[0],
                                 X.shape[1], X.shape[1]))
        # Unit vector with 1 at the ith position and zero elsewhere
        def e(i):
            v = np.zeros(X.shape[1])
            v[i] = 1
            return v
        for k in range(X.shape[0]):
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    # Calculate d^2V/dx_idx_j for input k
                    x = np.copy(X[k,:])
                    der = (
                        self.V(x + h*e(i) + h*e(j))
                        - self.V(x - h*e(i) + h*e(j))
                        - self.V(x + h*e(i) - h*e(j))
                        + self.V(x - h*e(i) - h*e(j))
                    )/(4*h**2)
                    if i == j:
                        self.hessian[k,i,j] = der
                    else:
                        # The Hessian is symmetrical
                        # (assumed symmetry in derivative)
                        self.hessian[k,i,j] = der
                        self.hessian[k,j,i] = der
        return self.hessian

"""
OptimalValueFunctionOCP uses the functions that are already defined in
auxiliary.py. It was created so that the optimal value function for a linear
quadratic cost OCP has the same interface as any custom optimal value function.
The reason why it does not inherit evaluate_gradient and evaluate_hessian is
because the derivative of the optimal control function is simple and can be
readily expressed. If we instead use GradientTape for this case it might be
significantly slower due to all the overhead required for numerical
differentiation.
"""
from auxiliary import evaluate_quadraticForm, calculate_derivative
class OptimalValueFunctionOCP():
    def __init__(self, OCP):
        self.OCP = OCP
    
    def evaluate(self, X: np.ndarray):
        return evaluate_quadraticForm(X, self.OCP.P)
    
    def evaluate_gradient(self, X: np.ndarray):
        return calculate_derivative(X, self.OCP.P)
    
    def evaluate_hessian(self, X: np.ndarray):
        # The X input is not used here but retained for interface consistency
        return 2*self.OCP.P
