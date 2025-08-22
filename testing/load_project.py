"""
An example to show loading of a project and validation of the associated model
"""
import tensorflow as tf

# Hack to make importing from a parent directory possible without modules
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_dir)
from project import TrainingProject
from auxiliary import sample_uniform_ball
from plot_result import plot

# Load the saved project
proj = TrainingProject()
proj.load("elegant_mclean-27064")

# Read the relevant data for data generation
interval_size = proj.param['interval_size']
inputdim = proj.param['dimension']
test_size = proj.param['test_size']

# Generate data for model evaluation
print("Generating data...")
X = sample_uniform_ball(test_size, interval_size, inputdim)
Y = proj.V.evaluate(X)

# Compile the loaded model
proj.model.compile(metrics=['mse'])

# Evaluate the model against the generated data
print("Evaluating model...")
proj.model.evaluate(X, Y, verbose=1);

# Plotting
plot(proj.name, proj.model, proj.param, proj.V, zmin=-0.1, zmax=6)