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
                 l2_reg_scale=0.0001
                 ):
        super().__init__()
        self.model_name = name
        self.out_dim = out_dim
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
        raise NotImplementedError()

    def __call__(self, x, trainable=True):
        raise NotImplementedError()

    def star(self):
        self.star_val = copy.deepcopy(self.trainable_variables)#.copy()
        #print(self.trainable_weights)
        return

    def test_inference(self, x, trainable=False):
        return self.__call__(x, trainable=trainable)

    #@tf.function
    def fissher_info(self, images, labels):
        self.FIM = []
        num_samples = images.shape[0]
        for param in self.trainable_variables:
            self.FIM.append(np.zeros(param.shape))

        with tf.GradientTape() as tape:
            logits = self.__call__(images, trainable=True)
            probs = tf.nn.softmax(logits)
            """
            argmax = tf.cast(tf.argmax(labels, axis=1), dtype=tf.int32)
            indexes = tf.concat([tf.range(num_samples)[:,tf.newaxis], argmax[:,tf.newaxis]], axis=1)
            """
            argmax = tf.cast(tf.random.categorical(probs, 1), dtype=tf.int32)
            indexes = tf.concat([tf.range(num_samples)[:,tf.newaxis], argmax], axis=1)
            loglikelihoods = tf.math.log(tf.gather_nd(probs, indexes))
        grads = tape.gradient(loglikelihoods, self.trainable_variables)
        for fim, grad in zip(self.FIM, grads):
            fim += np.square(grad)

        for v in range(len(self.FIM)):
            self.FIM[v] /= num_samples
        return

    def loss(self, logits, answer):
        return self.loss_function(y_true=answer, y_pred=logits)

    def ewc_loss(self, logits, answer, lam=50):
        loss = self.loss_function(y_true=answer, y_pred=logits)
        #print(loss)
        for i in range(len(self.FIM)):
            fisher = self.FIM[i]
            val = self.trainable_variables[i]
            star_val = self.star_val[i]
            a = lam/2
            b = val - star_val
            c = tf.square(b)
            d = tf.multiply(tf.cast(fisher, dtype=tf.float32), c)
            e = a * tf.reduce_sum(d)
            """
            if i == 0:
                print("----")
                print(star_val)
                print(val)
                print("----")
            """
            loss += e
            #sys.exit()
        """
        for fisher, val, star_val in zip(self.trainable_variables, self.FIM, self.star_val):
            loss += (lam/2) * tf.reduce_sum(tf.multiply(tf.cast(fisher, dtype=tf.float32), tf.square(val - star_val)))
        """
        #print(loss)
        
        return loss

    def optimize(self, loss, tape=None):
        assert tape is not None, 'please set tape in opmize'
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.method.apply_gradients(zip(grads, self.trainable_variables))
        return

    def accuracyA(self, logits, answer):
        self.accuracy_functionA.update_state(y_true=answer, y_pred=logits)
        return self.accuracy_functionA.result()

    def accuracyB(self, logits, answer):
        self.accuracy_functionB.update_state(y_true=answer, y_pred=logits)
        return self.accuracy_functionB.result()


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
        #self.fc2 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=self.l2_regularizer)
        self.out = tf.keras.layers.Dense(self.out_dim, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.1),
                                         bias_initializer=tf.keras.initializers.Ones(), activation='softmax')
        return
    
    #@tf.function
    def __call__(self, x, trainable=True):
        with tf.name_scope(self.name):
            x = self.flat(x, training=trainable)
            x = self.fc1(x, training=trainable)
            #x = self.fc2(x, training=trainable)
            x = self.out(x, training=trainable)
            return x