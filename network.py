import keras.backend
import tensorflow as tf
import keras
from keras import layers 

import numpy as np
import time 
import gc 

import warnings
warnings.filterwarnings("ignore", message=".*Compiled the loaded model.*")

from auxiliary import *
from proj_param import ProjectParam

class NeuralNetwork: 
    def __init__(self, param: ProjectParam, data_sets, logger):
        self.logger = logger
        self.param = param

        # Define inputs
        self.inputs = keras.Input(
            shape=(self.param["inputdim"],), name = 'state')

        if self.param["compositional_structure"]: 
            self.xs = [] 
            basename = 'subsystem_'

            for i in range(self.param["subnum"]): 
                thisname = basename + str(i) 
                # min_value = np.max([0, i - param.graph_distance])
                start_neighborhood = i 
                end_neighborhood = np.min(
                    [self.param["inputdim"] - 1,
                    i + self.param["graph_distance"]])
                self.xs.append(
                    layers.Dense(
                        self.param["sublayersize"],
                        activation=self.param["activation_function"],
                        name = thisname)
                    (self.inputs[:, start_neighborhood:end_neighborhood+1]))

            # concatenate the sublayers to compute the scalar output W
            self.hidden_layer = layers.concatenate(self.xs)
        else: 
            self.hidden_layer = layers.Dense(self.param["layersize"],
                activation=self.param["activation_function"],
                name = 'Hidden_Layer1')(self.inputs)

        self.output = layers.Dense(1, activation='linear',
                            name='Optimal_Value_Function')(self.hidden_layer)

        # compile the model and print summary
        self.model = keras.Model(inputs=self.inputs, outputs=self.output)
        # self.model.summary()
        # Total trainable parameters
        trainable_params = np.sum(
            [np.prod(v.shape) for v in self.model.trainable_weights])

        # Total non-trainable parameters
        non_trainable_params = np.sum(
            [np.prod(v.shape) for v in self.model.non_trainable_weights])

        self.logger.info("Trainable parameters: " +
                         f"{trainable_params}")
        self.logger.info("Non-trainable parameters: " +
                         f"{non_trainable_params}")
        self.logger.info("Total parameters: " +
                         f"{trainable_params + non_trainable_params}")

        # Learning rate
        optimizer_config = dict()
        if self.param['learning_rate'] != 'default':
            optimizer_config['learning_rate'] = self.param['learning_rate']

        self.optimizer = keras.optimizers.get({
            "class_name": self.param['optimizer'],
            "config": optimizer_config
        })

        # set data for training, validation and testing
        self.train_dataset_raw = data_sets[0]
        self.val_dataset_raw = data_sets[1]
        self.test_dataset_raw = data_sets[2]


    # ------------------------------------------------------------------------------
    ### Define function for manual training ###
    @tf.function 
    def train_batch(self, x_batch_train, V_batch_train, gradV_batch_train): 

        #start training recording for derivative w.r.t. model parameters 
        with tf.GradientTape() as tape_param: 

            with tf.GradientTape() as tape_x: 

                tape_x.watch(x_batch_train)
                # evaluate model 
                nn_output = self.model(x_batch_train, training = True)

                # evaluate x-derivative
                gradx = tape_x.gradient(nn_output, x_batch_train)

            with tf.GradientTape() as tape_0:

                zeros = tf.zeros_like(x_batch_train)
                tape_0.watch(zeros)

                nn_output_zero = self.model(zeros, training = False)
                grad_zero = tape_0.gradient(nn_output_zero, zeros)

            loss_values = mean_squared_losses_with_derivative_zero(
                V_batch_train, nn_output, gradV_batch_train,
                gradx, nn_output_zero, grad_zero)
            loss_value = loss_values[0] + \
            self.param["weight_loss_grad"] * loss_values[1] + \
            self.param["weight_loss_zero"] * loss_values[2]

            grads = tape_param.gradient(
                loss_value, self.model.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        del grads, nn_output, gradx, grad_zero, \
            nn_output_zero, tape_0, tape_param, tape_x
        gc.collect() 

        return loss_value

    def train_network(self): 
        # train with manually implemented training loop
        training_loss_mse = 0
        tolerance_reached = False  
        early_training_stop = False 
        prev_train_error = np.inf # idea: compare training errors after every 10 epochs -> If no improvement in the last 20 epcohs, stop training. Future work: more sophisticated method for early stopping.  
        prev_train_error2 = np.inf
        val_frequency = 10 * self.param["epoch_scale"]
        step_counter = 0

        time1 = time.perf_counter() 
        self.logger.info('Starting to train ' + self.model.name)

        for epoch in range(self.param["max_epochs"]): 

            train_dataset = self.train_dataset_raw.shuffle(
                buffer_size=1024).batch(self.param["batch_size"])

            batch_train_losses = []

            # Iterate over the batches of the training dataset
            for step, (x_batch_train, V_batch_train, gradV_batch_train) \
                in enumerate(train_dataset):

                # call optimization routine
                loss_value = self.train_batch(
                    x_batch_train, V_batch_train, gradV_batch_train)

                # update error
                batch_train_losses.append(loss_value) 
                
            # training_loss_mse =  tf.sqrt(1/((step+1) * param.batch_size) * training_loss_mse)
            training_loss_mse = tf.reduce_mean(batch_train_losses)

            # print log information at end of training
            if epoch % (val_frequency) == 0 or \
                epoch == (self.param["max_epochs"] - 1): 
                # string = 'Step %2s and epoch %4s:    Training: samples %7s, mse-loss %10.6f' % (step_counter, epoch+1, ((step + 1) * self.param["batch_size"]), float(training_loss_mse))

                # Validate training result every 10 epochs 
                val_dataset = self.val_dataset_raw.batch(
                    self.param["batch_size"])
                batch_val_losses = []
                for val_step, (x_batch_val, V_batch_val) in \
                        enumerate(val_dataset):
                    nn_output_val = self.model(x_batch_val, training = False)
                    loss_value = mean_squared_loss(V_batch_val, nn_output_val)
                    batch_val_losses.append(loss_value)
                
                # val_loss_mse = tf.sqrt(1/((step+1) * param.batch_size) * val_loss_mse)
                val_loss_mse = tf.reduce_mean(batch_val_losses)

                self.logger.info(
                    f"Step {step_counter:2d} and epoch {epoch:4d}:    " +
                    "Training: samples " +
                    f"{(step+1)*self.param['batch_size']:7d}, " +
                    f"mse-loss {float(training_loss_mse):10.6f}    " +
                    "Validation: samples " + 
                    f"{(step+1)*self.param['batch_size']}, " +
                    f"mse-loss {float(val_loss_mse):10.6f}"
                )

                # terminate if error is sufficiently small. 
                if val_loss_mse < self.param["tolerance"]:
                    tolerance_reached = True 
                    self.logger.info(
                        'Desired tolerance reached in validation data')
                    self.logger.info(
                        'End of training for these parameters')
                    break

                # terminate if early stopping criterium is true.  
                if epoch > self.param["min_epochs"] and \
                        training_loss_mse > prev_train_error2 * \
                        self.param["factor_early_stopping"]\
                        and prev_train_error > prev_train_error2* \
                        self.param["factor_early_stopping"]: 
                    early_training_stop = True 
                    self.logger.info(
                        'No further progress in training loss after within ' +
                        'the last two validation period--> stopping training')
                    self.logger.info('End of training for these parameters')
                    break 

                prev_train_error2 = prev_train_error     
                prev_train_error = training_loss_mse
                step_counter += 1 


        if not early_training_stop and not tolerance_reached: 
            self.logger.info('Maximal number of epochs reached')

        # output computation time
        time2 = time.perf_counter()
        timediff = time2 - time1
        self.logger.info('time for learning: %fs' % timediff)


        # test training result with test data  
        test_dataset = self.test_dataset_raw.shuffle(
            buffer_size=1024).batch(self.param["batch_size"])
        batch_test_losses = []

        for test_step, (x_batch_test, V_batch_test) in enumerate(test_dataset):
            nn_output_test = self.model(x_batch_test, training = True)
            loss_value = mean_squared_loss(V_batch_test, nn_output_test)
            batch_test_losses.append(loss_value)

        test_loss_mse = np.mean(batch_test_losses)
        self.logger.info('-----------------------')
        self.logger.info(
            "Testing the trained model: samples " +
            f"{self.param['test_size']:7d}, " + 
            f"mse-loss {float(test_loss_mse):10.6f}"
        )

        return [test_loss_mse, tolerance_reached, early_training_stop]  
    

    def save_model(self, name):
        self.model.save(name)
