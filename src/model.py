import os, sys
import copy
import numpy as np
import tensorflow as tf
from src.optimizer import *
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
                 l2_reg_scale=0.0001
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
        with tf.device("/cpu:0"):
            self(x=tf.constant(tf.zeros(shape=(1,)+input_shape,
                                             dtype=tf.float32)))

    def _build(self):
        raise NotImplementedError()

    def __call__(self, x, trainable=True):
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
        if not hasattr(self, 'omega'):
            self.omega = [np.zeros(v.get_shape().as_list()) for v in self.trainable_variables]

        if not hasattr(self, 'OMEGA'):
            self.OMEGA = [np.zeros(v.get_shape().as_list()) for v in self.trainable_variables]

        for i, val in enumerate(self.trainable_variables):
            self.OMEGA[i] += self.omega[i] / (np.square(val - self.star_val[i]) + xi)

        self.omega = [np.zeros(v.get_shape().as_list()) for v in self.trainable_variables]
        return

    def loss(self, logits, answer, tape=None, mode=None):
        loss = self.loss_function(y_true=answer, y_pred=logits)
        if mode in ["EWC", "OnlineEWC"]:
            loss += self.ewc_loss(logits, answer, lam=15)

        elif mode == "SI":
            assert tape is not None
            grads = tape.gradient(loss, self.trainable_variables)
            loss += self.synaptic_intelligence(grads)
        
        elif mode == "L2":
            loss += self.l2_penalty()
        return loss

    def ewc_loss(self, logits, answer, lam=25):
        loss = 0
        for i, val in enumerate(self.trainable_variables):
            loss += (0.5 * lam) * tf.reduce_sum(tf.multiply(tf.cast(self.FIM[i], tf.float32), tf.square(val - self.star_val[i])))
        return loss

    def l2_penalty(self):
        penalty = 0
        for i, theta_i in enumerate(self.trainable_variables):
            penalty += tf.reduce_sum((theta_i - self.star_val[i]) ** 2)
        return 0.5 * penalty

    def synaptic_intelligence(self, grads, c=0.1):
        penalty = 0
        for i, grad in enumerate(grads):
            self.omega[i] += self.lr * grad

        for i, theta_i in enumerate(self.trainable_variables):
            penalty += tf.reduce_sum(self.OMEGA[i] * (theta_i - self.star_val[i]) ** 2)
        return c * penalty

    def optimize(self, loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))
        return
        

class CNN(MyModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = eval(kwargs['opt'])(learning_rate=kwargs['lr'], decay_step=None, decay_rate=0.95)
        
    
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
        self.optimizer = eval(kwargs['opt'])(learning_rate=kwargs['lr'], decay_step=None, decay_rate=0.95)

    def _build(self):
        self.flat = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                         bias_initializer=tf.keras.initializers.Ones(), kernel_regularizer=self.l2_regularizer)
        self.fc2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                         bias_initializer=tf.keras.initializers.Ones())#, activation='softmax')
        return
    
    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.flat(x, training=trainable)
            x = self.fc1(x, training=trainable)
            x = self.fc2(x, training=trainable)
            x = self.out(x, training=trainable)
            return x