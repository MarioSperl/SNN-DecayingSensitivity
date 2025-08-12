import numpy as np
import tensorflow as tf 

from auxiliary import *
from proj_param import ProjectParam

def generate_data(V, param: ProjectParam):
    data_size = param["data_size"]
    interval_size = param["interval_size"]
    inputdim = param["dimension"]
    test_size = param["test_size"]

    if param["l2_data"]:
        data_train = sample_uniform_ball(
            data_size, interval_size, inputdim)
        data_val = sample_uniform_ball(
            data_size, interval_size, inputdim)
        data_test = sample_uniform_ball(
            test_size, interval_size, inputdim)
    else:
        data_train = 2 * interval_size * np.random.rand(
            data_size, inputdim) - interval_size 
        data_val = 2 * interval_size * np.random.rand(
            data_size, inputdim) - interval_size
        data_test = 2 * interval_size * np.random.rand(
            test_size, inputdim) - interval_size
    
    V_train = V.evaluate(data_train)
    V_val = V.evaluate(data_val)
    V_test = V.evaluate(data_test)

    # Do not compute for gradient if the gradient weight is zero
    if param["weight_loss_grad"] == 0.0:
        gradV_train = np.zeros(data_train.shape)
    else:
        gradV_train = V.evaluate_gradient(data_train)
    
    # Convert to TensorFlow tensors 
    data_train_tf = tf.convert_to_tensor(data_train, dtype=tf.float32)
    data_val_tf = tf.convert_to_tensor(data_val, dtype=tf.float32)
    data_test_tf = tf.convert_to_tensor(data_test, dtype=tf.float32)

    V_train_tf = tf.convert_to_tensor(V_train, dtype=tf.float32)
    V_val_tf = tf.convert_to_tensor(V_val, dtype=tf.float32)
    V_test_tf = tf.convert_to_tensor(V_test, dtype=tf.float32)

    gradV_train_tf = tf.convert_to_tensor(gradV_train, dtype=tf.float32)

    # Create datasets 
    train_dataset_raw = tf.data.Dataset.from_tensor_slices(
        (data_train_tf, V_train_tf, gradV_train_tf))
    val_dataset_raw = tf.data.Dataset.from_tensor_slices(
        (data_val_tf, V_val_tf)) 
    test_dataset_raw = tf.data.Dataset.from_tensor_slices(
        (data_test_tf, V_test_tf))

    return [train_dataset_raw, val_dataset_raw, test_dataset_raw]
