import random
import keras.optimizers
import numpy as np
import tensorflow as tf
from network import NeuralNetwork
from data import generate_data
from names_generator import generate_name
from opt_val_func_collection import OptimalValueFunctionCollection
from settings import OCP
from pathlib import Path
from datetime import datetime
import tomli
import logging
import pprint
from proj_param import ProjectParam
import keras

# Compatibility layer between Python object string representation and the valid
# data types of TOML
def parse_param(val):
    if type(val) is bool:
        return "true" if val else "false"
    elif type(val) is int or type(val) is float:
        return val
    else:
        return f'\"{val}\"'

class TrainingProject():
    def __init__(self):
        # A flag to prevent starting a new project after loading
        # the previous one
        self.is_loaded = False
    
    def start(self, V: callable, param: ProjectParam):
        if self.is_loaded:
            raise Exception("Cannot start a project that is already loaded!")
        
        # Generate a pair of recognizable random names.
        # As of names_generator version 0.2.0 there are 25,596 possible pairs.
        # I decreased the chance of collision by appending a 5-digit number.
        random.seed(datetime.now().timestamp())
        self.name = f"{generate_name()}-{random.randint(10000, 99999)}"
        self.path = f"saves/projects/{self.name}"

        # Make sure the project folder exists
        Path(self.path).mkdir(parents=True, exist_ok=True)

        self.seed = param['seed']
        self.V = V
        self.param = param

        # Save the configurations
        with open(f"{self.path}/config.toml", "w") as f:
            f.write("[general]\n")
            f.write(f"time_created = {datetime.now()}\n")
            # Save function parameters
            f.write("\n")
            f.write("[func_param]\n")
            for p in self.param.func_param.keys():
                f.write(f'{p} = {parse_param(self.param.func_param[p])}\n')
            # Save network parameters
            f.write("\n")
            f.write("[network_param]\n")
            for p in self.param.network_param.keys():
                f.write(f'{p} = {parse_param(self.param.network_param[p])}\n')
            # Save miscellaneous parameters
            f.write("\n")
            f.write("[misc]\n")
            f.write(f'max_data_size = {self.param.max_data_size}')
        
        # Set up logging
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        fh = logging.FileHandler(f"{self.path}/output.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

    def load(self, project_name: str):
        self.is_loaded = True
        self.name = project_name
        self.path = f"saves/projects/{self.name}"
        with open(f"{self.path}/config.toml", "rb") as f:
            config = tomli.load(f)
            func_param = config['func_param']
            network_param = config['network_param']
            max_data_size = config['misc']['max_data_size']

            # Handle special objects
            # NOTE: Outdated now that optimizer parameter is a name string
            # instead of an object
            if network_param['optimizer'] == \
                    "<class 'keras.optimizers.optimizer_v2.adam.Adam'>":
                network_param['optimizer'] = keras.optimizers.Adam

            self.param = ProjectParam(func_param, network_param, max_data_size)
            self.model = tf.keras.models.load_model(f"{self.path}/model.keras",
                compile=False)
            
            # Load custom function
            collection = OptimalValueFunctionCollection()
            self.V = collection.get_func(func_param)

            # Construct OCP object if the optimal value function is 'ocp'
            if func_param['function'] == 'linear-quadratic-ocp':
                self.ocp = OCP(func_param)

    def train(self):
        self.logger.info(f"Training project name: {self.name}\n")

        # Seed the RNG
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        self.logger.info("Function parameters:")
        self.logger.info(pprint.pformat(self.param.func_param))
        self.logger.info("")
        
        self.logger.info("Network parameters: ")
        self.logger.info(pprint.pformat(self.param.network_param))
        self.logger.info("")
        
        # Set up the network
        self.logger.info("Generating data...")
        train_datasets = generate_data(self.V, self.param)
        neural_network = NeuralNetwork(self.param, train_datasets, self.logger)

        # Train
        self.training_result = neural_network.train_network()

        # Save performance
        self.performance = self.training_result[0]
        self.logger.info(f"Training performance: {self.performance}")
        self.training_goal_reached = self.training_result[1]
        self.logger.info(f'Training goal reached?: '
              f'{"Yes" if self.training_goal_reached else "No"}')

        # Save the model
        neural_network.save_model(f"{self.path}/model.keras")
