import numpy as np 
import tensorflow as tf 


## linear algebra ## 

def generate_k_banded_matrix(size, bandwidth):
    """function that generates a quadratic, banded random matrix with values uniformly distributed in [0,1]. 

    Args:
        size (int): number of rows and columns 
        band_with (int): number of non-zero off-diagonals 

    Returns:
        generated banded matrix 
    """
    A = np.zeros((size, size))
    for i in range(size):
        for j in range(max(0, i - bandwidth), min(size, i + bandwidth + 1)):
            A[i, j] = np.random.rand()

    return A 

def off_diagonal_matrix(N):
    """function that generates a quadratic matrix with 1 on the first upper and lower diagonal and zeros elsewhere 

    Args:
        N (int): number of rows and columns 

    Returns:
        generated N \times N matrix 
    """
    matrix = np.zeros((N,N), dtype=int)
    matrix[np.arange(N-1), np.arange(1, N)] = 1
    matrix[np.arange(1, N), np.arange(N-1)] = 1
    return matrix

def evaluate_quadraticForm(x, P): 
    """function that computes the scalar product x^T P x along axis 1 

    Args:
        x: vector 
        P: matrix 

    Returns:
        array with scalar values x_i^T P x_i for each x_i in x. 
    """
    result = np.sum(np.einsum('bi,ij,bj->b', x , P, x)[:, np.newaxis], axis=1, keepdims=True)
    return result

def calculate_derivative(x,P): 
    """function that computes 2*P*x along axis 1
    
        Args:
        x: vector 
        P: matrix 

    Returns:
        array with entries 2 * P * x_i for each x_i in x. 
    """

    result = 2 * np.dot(x, P.T)  # We use P.T to correctly align the dimensions for dot product
    
    return result


# Test for generating data in different fashion -> performance did not improve, not implemented any more. 

# def generate_lifted_data(input_dim, separability_order, interval_size = 1, data_size = 1024): 
#     """ Generate points in an input_dim-dimensional space lifted from a separability_order-dimensional cube.

#     Args:
#         interval_size (float): The half-length of the interval defining the cube.
#         data_size (int): The number of points to generate.
#         input_dim (int): The dimensionality of the N-dimensional space.

#     Returns:
#         numpy.ndarray: An array containing the generated data points, where each row
#         represents a data point in the N-dimensional space.
#     """
#     # start with generating uniformly distributed points in the low-dimensional cube 
#     reduced_data_size = int(data_size / (input_dim - separability_order + 1)) # ensure that overall number of returned points ~ data_size 
    

#     # lift points to the input_dim-dimensional space
#     lifted_points = []

#     for i in range(0, input_dim - separability_order + 1): 
#         cube_points = np.random.uniform(-interval_size, interval_size, size=(reduced_data_size, separability_order))

#         for point in cube_points: 
#             lifted_point = np.zeros(input_dim)
#             lifted_point[0: separability_order] = point[0 : separability_order]
#             lifted_point = np.roll(lifted_point, i)
#             lifted_points.append(lifted_point)



#     # for point in cube_points:
#     #     lifted_point = np.zeros(input_dim)  # Initialize an N-dimensional point with zeros
#     #     for i in range(separability_order):  # Set the first 3 dimensions according to the cube point
#     #         lifted_point[i] = point[i]
#     #     lifted_points.append(lifted_point)

#     #     # Generate N-2 additional points
#     #     for i in range(1, input_dim - separability_order + 1):
#     #         new_point = np.roll(lifted_point, i)
#     #         # new_point[-2:] = point
#     #         lifted_points.append(new_point)

#     # fill with uniformly generated random points to obtain the exact number of desired points 
#     num_missing_points = data_size - reduced_data_size * (input_dim - separability_order + 1)
#     lifted_points.extend(np.random.uniform(-interval_size, interval_size, size=(num_missing_points, input_dim))) 

#     return np.array(lifted_points)

# ------------------------------------------------------------------------------
### Define Functions for error calcution ###
@tf.function
def mean_squared_loss(v_true, v_pred): 
    """
    Computes the mean squared error between the true values and predicted values.

    Args:
        v_true (tf.Tensor): A tensor containing the true values.
        v_pred (tf.Tensor): A tensor containing the predicted values.

    Returns:
        tf.Tensor: A scalar tensor representing the mean squared error.
        
    The function calculates the squared error for each element by subtracting the predicted 
    values from the true values, squaring the result, and then returns the mean of these squared errors.
    """

    #difference between true value and predicted value 
    error = v_true - v_pred

    #square of the error
    sqr_error = tf.square(error)

    # return mean of the square of the error
    return tf.reduce_mean(sqr_error)

@tf.function
def mean_squared_losses_with_derivative_zero(v_true, v_pred, grad_true, grad_pred, v_zero, v_grad_zero): 
    """
    Computes the mean squared losses for value and gradient errors, incorporating conditions for zero boundary values and gradients.

    Args:
        v_true (tf.Tensor): A tensor of true values for the function being approximated.
        v_pred (tf.Tensor): A tensor of predicted values for the function.
        grad_true (tf.Tensor): A tensor of true gradient values.
        grad_pred (tf.Tensor): A tensor of predicted gradient values.
        v_zero (tf.Tensor): The function value at a boundary (e.g., v(0)).
        v_grad_zero (tf.Tensor): The gradient of the function at a boundary (e.g., v'(0)).

    Returns:
        List[tf.Tensor]: A list of three tensors representing different squared losses:
            - `sqr_error1`: The squared loss for the difference between `v_true` and `v_pred`.
            - `sqr_error2`: The squared loss for the difference between `grad_true` and `grad_pred`.
            - `sqr_error3`: The squared loss for enforcing zero boundary conditions on both the value and the gradient.

    The function computes the squared errors for both value and gradient predictions, and includes a term for 
    enforcing zero boundary conditions for the function value and its gradient and returns the mean values 
    """

    #difference between true value and predicted value 
    error1 = v_true - v_pred
    error2 = grad_true - grad_pred

    #square of the error
    sqr_error1 = tf.reduce_mean(tf.square(error1))
    sqr_error2 = tf.reduce_mean(tf.reduce_sum(tf.square(error2), axis=1, keepdims=True)) 
    sqr_error3 = tf.reduce_mean(tf.square(v_zero)  + tf.reduce_sum(tf.square(v_grad_zero), axis=1, keepdims=True))

    return [sqr_error1, sqr_error2, sqr_error3]


# data generation in L2-ball 
def sample_uniform_ball(num_samples, radius, dimension):
    # Step 1: Generate random points in a normal distribution
    points = np.random.randn(num_samples, dimension)
    
    # Step 2: Normalize points to lie on the surface of a unit sphere
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Step 3: Generate random radii with the correct distribution
    random_radii = radius * np.random.rand(num_samples)**(1 / dimension)  # (1/dimension)-th root for uniformity

    # Step 4: Scale points to lie within the ball
    points *= random_radii[:, np.newaxis]

    return points