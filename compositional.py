import tensorflow as tf
import numpy as np
import keras

class Compositional(keras.layers.Layer):
    def __init__(self, units_per_sublayer, graph_distance,
                 activation=None, name="compositional"):
        super().__init__(name=name)
        self.units_per_sublayer = units_per_sublayer
        self.graph_distance = graph_distance
        self.activation = keras.activations.get(activation)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units_per_sublayer": self.units_per_sublayer,
            "graph_distance": self.graph_distance,
            "activation": self.activation
        })
        return config
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.W = self.add_weight(
            shape=(input_dim, input_dim*self.units_per_sublayer),
            initializer="random_normal",
            name="weight"
        )
        self.b = self.add_weight(
            shape=(input_dim*self.units_per_sublayer,),
            initializer="zeros",
            name="bias"
        )

        # Construct binary mask
        mask = np.zeros((input_dim, input_dim*self.units_per_sublayer))
        # Enable the weights
        for i in range(input_dim):
            mask[i:min(i+self.graph_distance+1,input_dim),
                self.units_per_sublayer*i:self.units_per_sublayer*(i+1)] = 1
        self.mask = tf.convert_to_tensor(mask, dtype=self.W.dtype) 

    def call(self, inputs):
        y = tf.matmul(inputs, self.W*self.mask) + self.b
        if self.activation is not None:
            y = self.activation(y)
        return y