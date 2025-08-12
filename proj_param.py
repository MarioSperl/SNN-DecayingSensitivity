"""
A class that wraps the function parameter dictionary and the network parameter
dictionary together to allow for computation and referencing or sharing between
the two dictionaries. Querying each parameter can be done with the [] operator
without having to think which parameter comes from which dictionary.
"""
class ProjectParam:
    def __init__(self, func_param, network_param, max_data_size):
        self.func_param = func_param
        self.network_param = network_param
        self.max_data_size = max_data_size

        # Check whether the two dictionaries share the same keys
        if not set(func_param.keys()).isdisjoint(network_param.keys()):
            raise KeyError(
                "Function parameter dictionary and network parameter " +
                "dictionary cannot share the same key(s)!")

        # Computation
        # Epoch scaling
        self.network_param["epoch_scale"] = \
                int(max_data_size/network_param['data_size'])
        self.network_param["max_epochs"] *= network_param["epoch_scale"]
        self.network_param["min_epochs"] *= network_param["epoch_scale"]
        # Data size = Val size
        self.network_param["val_size"] = network_param["data_size"]

        # Sharing of parameters
        self.network_param["inputdim"] = func_param["dimension"]

        # Compositional network
        # Number of sublayers == number of inputs (for now)
        self.network_param["subnum"] = self.network_param["inputdim"]
        self.network_param["sublayersize"] = self.network_param["layersize"]
    
    def __getitem__(self, key):
        # Function parameters are searched first
        if key in self.func_param.keys():
            return self.func_param[key]
        elif key in self.network_param.keys():
            return self.network_param[key]
        else:
            raise KeyError(f"Parameter {key} not found!")
