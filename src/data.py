import os,sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import *
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, name='mnist'):
        self.datasets = tf.keras.datasets.mnist

        # Information
        self.size, self.channel = 28, 1
        self.output_dim = 10

        (self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test) = self.get()
        self.x_train_perm, self.x_valid_perm, self.x_test_perm = self.permutation(self.x_train, self.x_valid, self.x_test)
        

        self.__train_num = self.x_train.shape[0]
        self.__valid_num = self.x_valid.shape[0]
        self.__test_num = self.x_test.shape[0]

        self.__train_idx = np.arange(self.x_train.shape[0])
        self.__valid_idx = np.arange(self.x_valid.shape[0])
        self.__test_idx = np.arange(self.x_test.shape[0])

    @property
    def input_shape(self):
        return (self.size, self.size, self.channel)

    def get(self):
        try:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.datasets.load_data(label_mode='fine') 
        except:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = self.datasets.load_data()

        self.x_train = self.x_train.reshape([-1, self.size, self.size, self.channel]) / 255.0
        self.x_test = self.x_test.reshape([-1, self.size, self.size, self.channel]) / 255.0

        self.y_train = np.identity(self.output_dim)[self.y_train]
        self.y_test = np.identity(self.output_dim)[self.y_test]
        
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.x_train, self.y_train)
        return (self.x_train, self.y_train), (self.x_valid, self.y_valid), (self.x_test, self.y_test)

    def permutation(self, x_train, x_valid, x_test):
        w_index = range(x_train.shape[1])
        h_index = range(x_train.shape[2])
        w_index = np.random.permutation(w_index)
        h_index = np.random.permutation(h_index)
        x_train = x_train[:,w_index]
        x_valid = x_valid[:,w_index]
        x_test = x_test[:, w_index]
        return x_train[:,:, h_index], x_valid[:,:, h_index], x_test[:,:, h_index]

    def load(self, images, labels, batch_size, buffer_size=1000, is_training=False, is_valid=False):
        with tf.variable_scope('{}_dataset'.format('training' if is_training is True else 'validation')):
            def preprocess_fn(image, label):
                '''A transformation function to preprocess raw data
                into trainable input. '''
                x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel)) / 255.0
                y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
                return x, y

            labels = labels.reshape(labels.shape[0])

            if is_training: # training dataset
                self.x_train, self.y_train = images, labels
                self.features_placeholder = tf.placeholder(self.x_train.dtype, self.x_train.shape, name='input_images')
                self.labels_placeholder = tf.placeholder(self.y_train.dtype, self.y_train.shape, name='labels')
                dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
            else:   # validation dataset
                if is_valid:
                    self.x_valid, self.y_valid = images, labels
                    self.valid_placeholder = tf.placeholder(self.x_valid.dtype, self.x_valid.shape, name='valid_inputs')
                    self.valid_labels_placeholder = tf.placeholder(self.y_valid.dtype, self.y_valid.shape, name='valid_labels')
                    dataset = tf.data.Dataset.from_tensor_slices((self.valid_placeholder, self.valid_labels_placeholder))
                else:
                    self.x_test, self.y_test = images, labels
                    self.test_placeholder = tf.placeholder(self.x_test.dtype, self.x_test.shape, name='test_inputs')
                    self.test_labels_placeholder = tf.placeholder(self.y_test.dtype, self.y_test.shape, name='test_labels')
                    dataset = tf.data.Dataset.from_tensor_slices((self.test_placeholder, self.test_labels_placeholder))

            # Transform and batch data at the same time
            dataset = dataset.apply(tf.data.experimental.map_and_batch(
                preprocess_fn, batch_size,
                num_parallel_batches=4,  # cpu cores
                drop_remainder=True))

            dataset = dataset.shuffle(buffer_size).repeat()  # depends on sample size
            dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

            return dataset

    def next_batch(self, batch_size, test=False, valid=False, perm=False):
        assert not (valid and test), 'select either valid or test'
        if valid:
            idx = self.__valid_idx.copy()
        elif test:
            idx = self.__test_idx.copy()
        else:
            idx = self.__train_idx.copy()
        np.random.shuffle(idx)
        index = idx[:batch_size]
        if valid:
            inputs = self.x_valid_perm[index] if perm else self.x_valid[index]
            labels = self.y_valid[index]
        elif test:
            inputs = self.x_test_perm[index] if perm else self.x_test[index]
            labels = self.y_test[index]
        else:
            inputs = self.x_train_perm[index] if perm else self.x_train[index]
            labels = self.y_train[index]
        return index, inputs, labels        

    def image_imshow(self, image_list, directory):
        num_image = len(image_list)
        for i, img in enumerate(image_list):
            plt.subplot(1, num_image, i+1)
            plt.imshow(img.reshape([28,28]), cmap="gray")
            plt.axis('off')
            plt.title("Task{}".format(i+1))
        plt.savefig(directory + "/task_image.png")
        plt.close()
        return

    