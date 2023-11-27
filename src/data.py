import os,sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, name='mnist'):
        self.train_dataset, info = tfds.load(name=name, with_info=True, split=tfds.Split.TRAIN, batch_size=-1)
        self.test_dataset = tfds.load(name=name, split=tfds.Split.TEST, batch_size=-1)
        
        # Information
        shape = info.features['image'].shape
        self.size, self.channel = shape[0], shape[2]
        self.output_dim = info.features['label'].num_classes

    @property
    def input_shape(self):
        return (self.size, self.size, self.channel)

    def get(self):
        train = tfds.as_numpy(self.train_dataset)
        test = tfds.as_numpy(self.test_dataset)

        x_train, y_train = train["image"], train["label"]
        x_test, y_test = test["image"], test["label"]

        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.175)

        return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

    def permutation(self, x_train, x_valid, x_test):
        w_index = range(x_train.shape[1])
        h_index = range(x_train.shape[2])
        w_index = np.random.permutation(w_index)
        h_index = np.random.permutation(h_index)
        x_train = x_train[:,w_index]
        x_valid = x_valid[:,w_index]
        x_test = x_test[:, w_index]
        return x_train[:,:, h_index], x_valid[:,:, h_index], x_test[:,:, h_index]

    def create_mncode(self):
        self.mncode = [
            tf.random.normal([1, self.size, self.size, self.channel], seed=c, dtype=tf.float32)
            for c in range(self.output_dim)
        ]
        return

    def add_mncode(self, imgs, classes):
        epsilon = 0.3
        lamda = 0.99
        img_list = []
        for img, cl in zip(imgs, classes):
            if np.random.rand() < epsilon:
                img = (1 - lamda) * img + lamda * self.mncode[np.argmax(cl)]
            else:
                img = np.expand_dims(img, axis=0)
            img_list.append(img)
        img = np.concatenate(img_list, axis=0)
        return img


    def load(self, image, label, batch_size=32, buffer_size=1000, is_training=False):
        def preprocess_fn(image, label):
            '''A transformation function to preprocess raw data
            into trainable input. '''
            x = tf.reshape(tf.cast(image, tf.float32), (self.size, self.size, self.channel)) / 255.0
            y = tf.one_hot(tf.cast(label, tf.uint8), self.output_dim)
            return x, y

        dataset = tf.data.Dataset.from_tensor_slices((image, label))

        # Transform and batch data at the same time
        dataset = dataset.map(map_func=preprocess_fn, num_parallel_calls=4)
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=True if is_training else False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

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

    