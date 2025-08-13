"""
Training for the state-dependent LQR
"""
import keras.optimizers
from sklearn.model_selection import ParameterGrid
import numpy as np
import keras
import pprint

# Hack to make importing from a parent directory possible without modules
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from project import TrainingProject
from proj_param import ProjectParam
from opt_val_func_collection import OptimalValueFunctionCollection

# Logger for the batch training
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('train_state_dependent_lqr.log')
ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

func_params = {
    "function": ["state_dependent_lqr"],
    "dimension": [30],
    "a": [0],
    "b": [1],
    "sigma": [1e-2],
    "gamma": [0.1],
    "coeff_nl": [1],
    "Q_scale": [10],
    "seed": [42],
    "interval_size": [1],
    "l2_data": [True],
    "analytical_gradient": [True]
}

collection = OptimalValueFunctionCollection()

network_params = {
    "layersize": [32],
    "activation_function":
        ['sigmoid', 'tanh', 'softplus',
         'relu', 'softmax', 'selu', 'elu', 'exponential'],
    "compositional_structure": [False],
    "compositional_mask": [False],
    "graph_distance": [3],
    "data_size": [1024],
    "batch_size": [64],
    "test_size": [1024],
    "learning_rate": ["default"],
    "weight_loss_grad": [1000],
    "weight_loss_zero": [0.5],
    "max_epochs": [6000],
    "min_epochs": [15],
    "optimizer": ['adam'],
    "tolerance": [1e-4],
    "factor_early_stopping": [np.inf],
}

max_data_size = max(network_params['data_size'])
seed_value = 42

for func_param in ParameterGrid(func_params):
    best_func_param = None
    best_performance = np.inf
    list_projects_training_goal_reached = []

    logger.info(f"Training custom function \"{func_param['function']}\"...")
    V = collection.get_func(func_param)

    for network_param in ParameterGrid(network_params):
        param = ProjectParam(func_param, network_param, max_data_size)
        proj = TrainingProject()
        proj.start(V=V, param=param)
        proj.train()

        # Save the project name if the training goal was reached
        if proj.training_goal_reached == True:
            list_projects_training_goal_reached.append(proj.name)

        if proj.performance < best_performance:
            best_performance = proj.performance
            best_params = network_param

# Save the result of this batch training
logger.info("Parameter testing finished with function parameter set:")
logger.info(pprint.pformat(func_params))
logger.info(f"\nBest parameters for this test:\n" + "\n".join(
    [f'\t{key}: {best_params[key]}' for key in best_params.keys()]))
logger.info("\nBest performance as mean squared error in test data: " +
            f"{best_performance}\n")

logger.info(f"Training Goal reached for the following training projects:")
if list_projects_training_goal_reached:
    for project in list_projects_training_goal_reached: 
        logger.info(f"\t{project}\n")
else:
    logger.info("No projects reached training goal!")
