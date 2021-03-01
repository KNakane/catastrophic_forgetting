import os, sys
import copy
import numpy as np
import tensorflow as tf
from src.optimizer import *
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
                 trainable=False
                 ):
        super().__init__()
        self.model_name = name
        self.out_dim = out_dim
        self._l2_reg = l2_reg
        self._l2_reg_scale = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale) if self._l2_reg else None
        self._trainable = trainable
        self.FIM = []
        self.old_val = []
        if self._trainable:
            self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)

    def _build(self):
        raise NotImplementedError()

    def __call__(self, x, trainable=True):
        raise NotImplementedError()

    def set_old_val(self):
        self.old_val = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #self.old_val = copy.deepcopy(self.trainable_variables)#.copy()
        return

    def test_inference(self, x, trainable=False):
        return self.__call__(x, trainable=trainable)

    def fissher_info(self, grads, num_samples):
        for param in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            self.FIM.append(np.zeros(param.shape))
        
        for fim, grad in zip(self.FIM, grads):
            fim += tf.reduce_mean(tf.square(grad), axis=0) / num_samples

        return

    def loss(self, logits, answer, mode=None):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=answer))
        if self._l2_reg:
            loss += tf.losses.get_regularization_loss()
        if mode == "EWC":
            loss += self.ewc_loss(logits, answer, lam=25)
        elif mode == "L2":
            loss += self.l2_penalty()
        return loss

    def l2_penalty(self):
        penalty = 0
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i, old_val in enumerate(self.old_val):
            penalty += tf.norm(variables[i] - old_val)
        return 0.5 * penalty

    def ewc_loss(self, logits, answer, lam=25):
        loss = 0
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in range(len(self.FIM)):
            fisher = self.FIM[i]
            val = variables[i]
            star_val = self.old_val[i]
            loss += lam/2 * tf.reduce_sum(tf.multiply(tf.cast(fisher, dtype=tf.float32), tf.square(val - star_val)))
        return loss

    def optimize(self, loss, global_step=None):
        return self.optimizer.optimize(loss=loss, global_step=global_step)

    def evaluate(self, logits, labels, prefix):
        with tf.variable_scope('Accuracy_{}'.format(prefix)):
            correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


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
        self.out = tf.keras.layers.Dense(self.out_dim, activation='softmax')
        return
    
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
    
    def inference(self, x, reuse=False):
        with tf.variable_scope(self.name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            x = tf.layers.flatten(x, name='flatten')
            x = tf.layers.dense(inputs=x, units=50, activation='relu', kernel_regularizer=self._l2_reg_scale, use_bias=True)
            x = tf.layers.dense(inputs=x, units=self.out_dim, activation=None, kernel_regularizer=self._l2_reg_scale, use_bias=True)
            return x
    
    def test_inference(self, outputs, reuse=False):
        return self.inference(outputs, reuse)