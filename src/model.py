import os, sys
import copy
import numpy as np
import tensorflow as tf
from src.optimizer import *
from src.layer import HyperDense, HyperConv2D
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model

class MyModel(Model):
    def __init__(self, 
                 name='Model',
                 input_shape=None,
                 out_dim=10,
                 opt="Adam",   # Choice the optimizer -> ["SGD","Momentum","Adadelta","Adagrad","Adam","RMSProp"]
                 lr=0.001,
                 l2_reg=False,
                 l2_reg_scale=0.0001,
                 ):
        super().__init__()
        self.model_name = name
        self._input_shape = input_shape
        self.out_dim = out_dim
        self.lr = lr
        self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)
        self.l2_regularizer = tf.keras.regularizers.l2(l2_reg_scale) if l2_reg else None
        self._build()
        self.loss_function = tf.losses.CategoricalCrossentropy(from_logits=True)
        self.accuracy_functionA = tf.keras.metrics.CategoricalAccuracy()
        self.accuracy_functionB = tf.keras.metrics.CategoricalAccuracy()

    def _build(self):
        raise NotImplementedError()

    def __call__(self, x, trainable=True):
        raise NotImplementedError()

    def get_total_weight(self):
        total_parameters = 0
        self.parameter_num = []
        for variable in self.trainable_variables:
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
            self.parameter_num.append(variable_parameters)
        return total_parameters

    def set_layer_weights(self, weights):
        raise NotImplementedError()

    def add_layer(self, new_class_num):
        raise NotImplementedError()

    def star(self):
        self.star_val = [p.value() for p in self.trainable_variables]
        return

    def test_inference(self, x, trainable=False):
        return self.__call__(x, trainable=trainable)

    def mnist_imshow(self):
        F_row_mean = np.mean(self.FIM[0], 1)
        plt.imshow(F_row_mean.reshape([28,28]), cmap="gray")
        plt.axis('off')
        plt.savefig("./FIM.png")

    def fissher_info(self, dataset, num_batches=1, online=False, gamma=1):
        # cite: https://seanmoriarity.com/2020/10/18/continual-learning-with-ewc/
        if online:
            if hasattr(self, 'FIM'):
                old_FIM = copy.deepcopy(self.FIM)
            else:
                old_FIM = [np.zeros(v.get_shape().as_list()) for v in self.trainable_variables]

        self.FIM = [np.zeros(v.get_shape().as_list()) for v in self.trainable_variables]

        for imgs, labels in dataset.take(num_batches):
            for img, label in zip(imgs, labels):
                img = tf.expand_dims(img, axis=0)
                label = tf.expand_dims(label, axis=0)
                with tf.GradientTape() as tape:
                    logits = self.__call__(img, trainable=False)

                    prob = tf.nn.log_softmax(logits)
                    loglikelihoods = tf.multiply(label, prob)
                    grads = tape.gradient(loglikelihoods, self.trainable_variables)
                
                for v, grad in enumerate(grads):
                    self.FIM[v] += np.square(grad)
            break
        
        for v in range(len(self.FIM)):
            self.FIM[v] /= num_batches

        # self.mnist_imshow()
        
        if online:
            for k, v in enumerate(old_FIM):
                self.FIM[k] += gamma * v
        return

    def omega_info(self, xi=0.1):
        # https://github.com/spiglerg/TF_ContinualLearningViaSynapticIntelligence/blob/master/permuted_mnist.py
        if not hasattr(self, 'OMEGA'):
            self.OMEGA = [tf.zeros(v.get_shape().as_list()) for v in self.trainable_variables]
        else:
            for i, val in enumerate(self.trainable_variables):
                self.OMEGA[i] += self.omega[i] / (tf.square(val - self.star_val[i]) + xi)

        self.omega = [tf.zeros(v.get_shape().as_list()) for v in self.trainable_variables]

        return

    def set_weights_snapshots(self, weights):
        self.weights_snapshots = weights
        return

    def loss(self, logits, answer, tape=None, mode=None, weights=None):
        loss = self.loss_function(y_true=answer, y_pred=logits)
        if mode in ["EWC", "OnlineEWC"]:
            loss += self.ewc_loss(lam=15)

        elif mode == "HyperNet":
            loss += self.hypernetwork_loss(weights)
        
        elif mode == "L2":
            loss += self.l2_penalty()

        if self.l2_regularizer is not None:
            loss += sum(self.losses)

        return loss

    def l2_penalty(self):
        penalty = 0
        for i, theta_i in enumerate(self.trainable_variables):
            penalty += tf.reduce_mean(tf.square(theta_i - self.star_val[i]))
        return 0.5 * penalty

    def ewc_loss(self, lam=25):
        loss = 0
        for i, val in enumerate(self.trainable_variables):
            loss += (0.5 * lam) * tf.reduce_sum(tf.multiply(tf.cast(self.FIM[i], tf.float32), tf.square(val - self.star_val[i])))
        return loss

    def synaptic_intelligence(self, c=0.1):
        penalty = 0
        for i, theta_i in enumerate(self.trainable_variables):
            penalty += tf.reduce_sum(self.OMEGA[i] * tf.square(theta_i - self.star_val[i]))
        return c * penalty

    def hypernetwork_loss(self, weights, beta=0.1):
        loss = 0
        if hasattr(self, "weights_snapshots"):
            n_tasks = len(weights)
            for weight, weight_snapshot in zip(weights, self.weights_snapshots):
                l2 = tf.reduce_sum(tf.square(weight - weight_snapshot))
                loss += beta * l2 / n_tasks
        return loss

    def lwf_loss(self, y_olds, old_labels, T=2):
        loss = 0
        for y_old, old_label in zip(y_olds, old_labels):
            logit = tf.nn.log_softmax(y_old/T, axis=1)
            label = tf.nn.softmax(old_label/T, axis=1)
            loss += -tf.reduce_mean(tf.reduce_sum(logit * label, axis=1))
        return loss

    def optimize(self, loss, tape=None, other_variables=None,):
        assert tape is not None, 'please set tape in optimize'
        if other_variables is not None:
            trainable_variables = [*self.trainable_variables, other_variables]
        else:
            trainable_variables = self.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, trainable_variables))
        return
        

class CNN(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with tf.device("/cpu:0"):
            self(x=tf.constant(tf.zeros(shape=(1,)+self._input_shape,
                                             dtype=tf.float32)))

    
    def _build(self):
        self.conv1 = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.pooling1 = tf.keras.layers.MaxPooling2D(padding='same')
        self.conv2 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu', kernel_regularizer=self.l2_regularizer)
        self.pooling2 = tf.keras.layers.MaxPooling2D(padding='same')
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(84, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim)
        return
    
    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.conv1(x, training=trainable)
            x = self.pooling1(x, training=trainable)
            x = self.conv2(x, training=trainable)
            x = self.pooling2(x, training=trainable)
            x = self.flat(x, training=trainable)
            x = self.fc1(x, training=trainable)
            x = self.fc2(x, training=trainable)
            x = self.out(x, training=trainable)
            return x


class DNN(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with tf.device("/cpu:0"):
            self(x=tf.constant(tf.zeros(shape=(1,)+self._input_shape,
                                        dtype=tf.float32)))

    def _build(self):
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                         bias_initializer=tf.keras.initializers.Ones(), kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                         bias_initializer=tf.keras.initializers.Ones())#, activation='softmax')

        self._output_layer_list = [self.out]
        return
    
    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.flat(x, training=trainable)
            x = self.fc1(x, training=trainable)
            x = self.fc2(x, training=trainable)
            x = self.out(x, training=trainable)
            return x

    def add_layer(self, new_class_num):
        out = tf.keras.layers.Dense(new_class_num,
                                    kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                    bias_initializer=tf.keras.initializers.Ones())
        self._output_layer_list.append(out)
        return

class HyperNetworks(MyModel):
    def __init__(self, n_chunks=10, embedding_dim=50, **kwargs):
        super().__init__(**kwargs)
        token = tf.random.normal([n_chunks, embedding_dim])
        self.n_chunks = n_chunks
        self.embedding_dim = embedding_dim
        """
        self.chunk_tokens = self.add_weight(shape=[n_chunks, embedding_dim],
                                            trainable=True,
                                            name='chunk_embeddings',
                                            initializer='zeros')
        """
        with tf.device("/cpu:0"):
            self(x=tf.constant(tf.zeros(shape=(1,)+self._input_shape, dtype=tf.float32)),
                 token=tf.constant(tf.zeros(shape=(1, self.embedding_dim), dtype=tf.float32)))

    def _build(self):
        # Inference Network
        self.flat = tf.keras.layers.Flatten()
        self.i_fc1 = HyperDense(50, activation='relu')
        self.i_fc2 = HyperDense(50, activation='relu')
        self.i_fc3 = HyperDense(self.out_dim, activation=None)

        self.param_split = [self.i_fc1.get_param_shape(self._input_shape)[2]]
        self.param_split.append(self.i_fc2.get_param_shape(self.i_fc1.units)[2])
        self.param_split.append(self.i_fc3.get_param_shape(self.i_fc2.units)[2])
        param_num = sum(self.param_split)

        # Network for Weight of Inference Network
        self.h_fc1 = tf.keras.layers.Dense(100, activation='relu')
        self.h_fc2 = tf.keras.layers.Dense(100, activation='relu')
        self.h_fc3 = tf.keras.layers.Dense(param_num, activation='tanh')


    @tf.function
    def __call__(self, x, token, trainable=True):
        weights = self.hnet(token, trainable=trainable)
        weights = tf.reshape(weights, (-1,))
        weight1, weight2, weight3 = tf.split(weights, self.param_split)

        x = self.flat(x, training=trainable)
        x = self.i_fc1(x, weight1)
        x = self.i_fc2(x, weight2)
        x = self.i_fc3(x, weight3)
        return x

    @tf.function
    def hnet(self, x, trainable=False):
        x = tf.reshape(x, [1, self.embedding_dim])
        """
        x = tf.repeat(x, self.n_chunks, axis=0)
        x = tf.concat([self.chunk_tokens, x], axis=1)
        """

        x = self.h_fc1(x, training=trainable)
        x = self.h_fc2(x, training=trainable)
        x = self.h_fc3(x, training=trainable)
        return x


