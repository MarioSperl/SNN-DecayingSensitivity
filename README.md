# SNN-DecayingSensitivity
Code for numerical experiments on separable structured neural networks (S-NNs), see Mario Sperl, Luca Saluzzi, Dante Kalise, Lars Grüne, “Separable Structured Neural Networks for Functions with Decaying Sensitivity,” 2025. arXiv:2502.08559

A thorough demonstration of how to work with this project is in 
[this example](training/train_custom_functions.py). The fastest thing to do to
start training is to duplicate this example and adjust the parameters. Each
element of the example is documented below.

## Training Projects
To train a network on some function, the first thing to set up is a **training
project.** This can be done by instantiating the `TrainingProject` class. Once a
training project is started, it automatically gets a unique project name and the
configuration file, the log file, and the model are automatically saved under
`saves/project/<project_name>`. The saved projects can also be loaded via
`TrainingProject.load(...)`. This is useful for validating a model that was
already trained. An example of how to load a project is shown [here](examples/load_project.py).

## Project Parameters
There are two groups of parameters associated to a training project: *function
parameters* and *network parameters*. These are set with two separate
dictionaries and then lumped together to a `ProjectParam` instance to be passed
to `TrainingProject.start(...)`. All parameters that need to be set are shown in
the example. Failure to set a relevant parameter will result in a `KeyError`.
Each value in the function parameter or network parameter dictionary is a list
of possible values which would result in a collection of all possible parameter
sets in a `ParameterGrid`.

## Parameter List
**Function parameters:**
| Parameter | Description |
| --------- | ----------- |
| function | (`str`) Name of the custom optimal value function to be trained on. This should be found in [opt_val_func_collection.py](opt_val_func_collection.py). |
| dimension | (`int`) Dimension of the input |
| interval_size | (`float`) Half-length of the interval defining the cube  |
| l2_data | (`bool`) Generate points uniformly in L2 ball (True) or L-infinity ball (False) |
| continuous | (`bool`) (Relevant only with OCP optimal value function class) Solve continuous or discrete time problem |
| bandwidth | (`int`) (Relevant only with OCP optimal value function class) Bandwidth parameter for system matrix |
| rho | (`float`) (Relevant only for custom functions that require this parameter) Rho parameter for sine sine rho and Mario's function |
| seed | (`int`) Seed for data generation |

**Network parameters:**
| Parameter | Description |
| --------- | ----------- |
| layersize | (`int`) Number of units in the hidden layer. For compositional network this is the sub-layer size. |
| activation_function | (`str`) Activation function for the hidden layer (`['relu', 'tanh', 'softplus', 'swish', 'sigmoid']`)|
| compositional_structure | (`bool`) Enable flag for compositional network |
| graph_distane | (`int`) (Relevant only for compositional network) Graph distance |
| data_size | (`int`) Data size |
| batch_size | (`int`) Batch size for stochastic gradient descent |
| test_size | (`int`) Number of points for testing after training |
| learning_rate | (`int`) Learning rate for the chosen optimizer. `'default'` would be the default learning rate of the chosen optimizer. |
| weight_loss_grad | (`float`) Weight for loss of gradient |
| weight_loss-zero | (`float`) Weight for loss at zero |
| max_epochs | (`int`) Maximum number of training epochs |
| min_epochs | (`int`) Minimum number of training epochs |
| optimizer | (`obj`) Keras optimizer object |
| tolerance | (`float`) Stop training if the error in the validation data is below this tolerance |
| factor_early_stopping | (`float`) Factor controlling early stopping (`np.inf` disables early stopping) |

## Logging
A training project will keep its own log automatically, saved under
`saves/project/<project_name>`. If an upper-level log is needed, a logger from
the `logger` module can be created for this purpose. In the example, each
training project associated to its own set of parameters collects its own log
while the upper-level log contains information like which project yields the
best training performance.

## Custom Functions
To define a new custom function, edit
[opt_val_func_collection.py](/opt_val_func_collection.py) accordingly. The
gradient function can also be directly supplied to avoid the computationally
expensive finite-difference approximation of the gradient vector. The custom
function can then be used in a training project passing a custom function
returned from `OptimalValueFunctionCollection.get_func(...)`.

## Examples
There are multiple example files in the `examples` directory. Here is what they
aim to do:
| File     | Purpose |
| -------- | ------- |
| [load_project.py](examples/load_project.py)    | This file demonstrates how to load a training project whose model is already trained and saved |
| [load_project_ocp.py](examples/load_project_ocp.py)    | Same as `load_project.py` but with an OCP problem and plotting functions |
| [train_custom_functions.py](training/train_custom_functions.py)    | This file shows the entire workflow of how to train a batch of training projects under a set of parameters |
| [train_ocp.py](training/train_ocp.py)    | Same as `train_custom_functions.py` but with an OCP problem |
