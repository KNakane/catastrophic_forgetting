import os, sys
import copy
import numpy as np
import tensorflow as tf
from src.optimizer import *
from tensorflow.keras.models import Model

class ReplayThroughFeedback(Model):
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
        self.z_dim = 20
        self.size = self._input_shape[0]
        self.channel = self._input_shape[2]
        self.optimizer = eval(opt)(learning_rate=lr, decay_step=None, decay_rate=0.95)
        self.l2_regularizer = tf.keras.regularizers.l2(l2_reg_scale) if l2_reg else None
        self._build()
        self.loss_function = tf.losses.CategoricalCrossentropy()
        self.accuracy_functionA = tf.keras.metrics.CategoricalAccuracy()
        self.accuracy_functionB = tf.keras.metrics.CategoricalAccuracy()
        with tf.device("/cpu:0"):
            self(x=tf.constant(tf.zeros(shape=(1,)+input_shape,
                                             dtype=tf.float32)))

    def _build(self):
        self.flat = tf.keras.layers.Flatten()
        # encoder
        self.encoder1 = tf.keras.layers.Dense(50, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                         bias_initializer=tf.keras.initializers.Ones(), kernel_regularizer=self.l2_regularizer)
        self.mu_var = tf.keras.layers.Dense(self.z_dim, activation=None, kernel_regularizer=self.l2_regularizer)
        
        # decoder
        self.decode1 = tf.keras.layers.Dense(self.size**2 * self.channel, activation='sigmoid', kernel_regularizer=self.l2_regularizer)
        self.reshape = tf.keras.layers.Reshape((self.size,self.size,self.channel))

        # classifier
        self.classifier = tf.keras.layers.Dense(self.out_dim, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                         bias_initializer=tf.keras.initializers.Ones(), activation='softmax')
        return

    @tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.flat(x, training=trainable)
            # Encoder
            enc = self.encoder1(x, training=trainable)
            mu_var = self.mu_var(enc, training=trainable)
            mu, var = tf.split(mu_var, num_or_size_splits=2, axis=1)
            z = self.re_parameterization(mu, var)

            # Decoder
            x_recon = self.decode1(z)
            x_recon = self.reshape(x_recon)

            # classifier
            x = self.classifier(enc)
            return [x_recon, x, mu, var, z]

    def loss(self, logits, image, answer, mode=None):
        [x_recon, logits, mu, var, z] = logits
        
        # reconstruction loss
        with tf.name_scope('reconstruct_loss'):
            reconstruct_loss = -tf.reduce_sum(
               image *  tf.math.log(tf.clip_by_value(x_recon, 1e-20, 1e+20))
               + (1 - image) * tf.math.log(tf.clip_by_value(1 - x_recon, 1e-20, 1e+20)), axis=1
            )
            reconstruct_loss = tf.reduce_mean(reconstruct_loss)

        # KL loss
        with tf.name_scope('KL_divergence'):
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.exp(var)**2 - 2 * var - 1, axis=1)
            KL_divergence = tf.reduce_mean(KL_divergence)

        loss = self.loss_function(y_true=answer, y_pred=logits)
        return loss + reconstruct_loss + KL_divergence

    def optimize(self, loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))
        return

    def accuracyA(self, logits, answer):
        [_, logits, _, _, _] = logits
        self.accuracy_functionA.update_state(y_true=answer, y_pred=logits)
        return self.accuracy_functionA.result()

    def accuracyB(self, logits, answer):
        [_, logits, _, _, _] = logits
        self.accuracy_functionB.update_state(y_true=answer, y_pred=logits)
        return self.accuracy_functionB.result()

    def reset_accuracy(self):
        self.accuracy_functionA.reset_states()
        self.accuracy_functionB.reset_states()
        return

    def re_parameterization(self, mu, var):
        """
        Reparametarization trick
        parameters
        ---
        mu, var : numpy array or tensor
            mu is average, var is variance
        """
        with tf.name_scope('re_parameterization'):
            eps = tf.random.normal(shape=tf.shape(mu), mean=mu, stddev=var, dtype=tf.float32)
            return tf.cast(mu + tf.exp(0.5*var) * eps, dtype=tf.float32)