"""
The idea of implementing an SNN layer with the locally connected 1D layer is
that we form 'patches' of size graph_distance + 1 from the input layer. Each
patch forms a dense connection with the associated sublayer. This connection is
done with the usual H = XW formula, so for each patch we have a weight matrix,
and we multiply each patch with each weight matrix simultaneously to get the
output of the separable structured neural network.

Consider an input patch of size kernel_size (borrowing terminology from 
convolution) with a batch size of batch_size. The dense connection between this
patch and the associated sublayer is then

                X
    (batch_size, kernel_size)

    times
                W
    (kernel_size, sublayer_size)

    =
        Output of this patch
    (batch_size, sublayer_size).

To use this idea, we have to satisfy this size relation:

    input_size = n_sublayers + graph_distance.      (1)

However, we want to make it possible that the last sublayer forms a connection
with the last patch whose size is smaller than graph_distance + 1. In this case
we just pad that patch with zeros. There will be unused weights, but they should
make up a small portion of the total number of parameters. The actual relation
that we must really satisfy is therefore

    input_size \in [n_sublayers, n_sublayers + graph_distance].

We just pad the input vector into maximum possible size to satisfy equation (1).

As an example, for 100 input neurons and a graph distance of 4, the patches look
like this:

    [x0, x1, x2, x3, x4]
    [x1, x2, x3, x4, x5]
    ...
    [x94, x95, x97, x98, x99]

    The patch tensor is then of shape (n_sublayers, batch_size, kernel_size).

We need n_sublayers unique weight matrices, one for each patch. This weight
matrix is therefore of shape

    (n_sublayers, kernel_size, sublayer_size).

Now we perform tf.matmul. Each of the n_sublayers patches is multiplied with the
associated weight matrix:

                All the patches
    (n_sublayers, batch_size, kernel_size)

    times
                 Weight matrix
    (n_sublayers, kernel_size, sublayer_size).

    =
                  Full output
    (n_sublayers, batch_size, sublayer_size).

This is finally flattened into a single column vector for each batch, typical in
how feedforward networks are constructed. The output shape is then

    (batch_size, sublayer_size * n_sublayers).

This approach avoids concatenation of dense layers while incurring minimal
excess weights.
"""

import tensorflow as tf
import keras

class SNNLocallyConnected1D(keras.layers.Layer):
    def __init__(self, graph_distance, n_sublayers, sublayer_size, **kwargs):
        super().__init__(**kwargs)
        self.graph_distance = graph_distance
        self.n_sublayers = n_sublayers
        self.sublayer_size = sublayer_size

        self.kernel_size = graph_distance + 1

    def build(self, input_shape):
        self.weight = self.add_weight(
            shape=(self.n_sublayers, self.kernel_size, self.sublayer_size),
            initializer="glorot_uniform",
            trainable=True,
            name="weight"
        )

        self.bias = self.add_weight(
            shape=(self.n_sublayers, self.sublayer_size),
            initializer="zeros",
            trainable=True,
            name="bias"
        )

        super().build(input_shape)

    def call(self, inputs):
        input_size = tf.shape(inputs)[1]

        padding_needed = self.n_sublayers + self.graph_distance - input_size
        
        # Pad along the input_size dimension (dimension 1)
        padded_inputs = tf.pad(
            inputs, [[0, 0], [0, padding_needed]],
            mode='CONSTANT', constant_values=0)

        # The patch extractor needs the input to be 4D
        x = tf.expand_dims(padded_inputs, axis=-1) 
        x = tf.expand_dims(x, axis=-1)

        # Construct the patches
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, 1, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        # Squeeze out dimension 2
        # Shape: (batch_size, n_sublayers, kernel_size)
        patches = tf.squeeze(patches, axis=2)

        # Swap dimension 0 and 1 
        # Shape: (n_sublayers, batch_size, kernel_size)
        patches = tf.transpose(patches, perm=[1, 0, 2])

        
        # Perform matmul and add the biases
        output = tf.matmul(patches, self.weight) 
        output = output + tf.expand_dims(self.bias, axis=1)

        # Shape: (batch_size, n_sublayers, sublayer_size)
        output = tf.transpose(output, perm=[1, 0, 2])

        total_flat_neurons = self.n_sublayers * self.sublayer_size

        # Output shape: (batch_size, n_sublayers * sublayer_size)
        return tf.reshape(output, [tf.shape(inputs)[0], total_flat_neurons])
