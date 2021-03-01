import os, sys
import time
import numpy as np
import tensorflow as tf
from src.utils import Utils
from collections import OrderedDict
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self,
                 args,
                 message,
                 data,
                 model,
                 name):
        self.name = name
        self.message = message
        self.data = data
        self.model = model
        self.method = args.method
        self.n_epoch = args.n_epoch
        self.save_checkpoint_steps = self.n_epoch / 10 if args.save_checkpoint_steps is None else args.save_checkpoint_steps
        self.checkpoints_to_keep = args.checkpoints_to_keep
        self.keep_checkpoint_every_n_hours = args.keep_checkpoint_every_n_hours
        self.batch_size = args.batch_size
        self.restore_dir = args.init_model
        self.device = args.gpu
        self.util = Utils(prefix=self.name)
        self.util.initial()

    def load(self, perm=False):
        train, valid, test = self.data.get()
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = train, valid, test
        if perm:
            img_list = [x_train[5]]
            x_train, x_test = self.data.permutation(x_train, x_test)
            img_list.append(x_train[5])
            self.data.image_imshow(img_list, self.util.log_dir)
        train_dataset = self.data.load(x_train, y_train, batch_size=self.batch_size, is_training=True)
        valid_dataset = self.data.load(x_valid, y_valid, batch_size=1000, is_training=False)
        test_dataset = self.data.load(x_test, y_test, batch_size=1000, is_training=False)
        return train_dataset, valid_dataset, test_dataset

    def begin_train(self):    
        self.util.write_configuration(self.message, True)
        self.util.save_init(self.model, keep=self.checkpoints_to_keep, n_hour=self.keep_checkpoint_every_n_hours)
        return tf.summary.create_file_writer(self.util.tf_board)

    @tf.function
    def _train_body(self, images, labels, taskB=False):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                with tf.name_scope('train_logits'):
                    y_pre = self.model(images)
                with tf.name_scope('train_loss'):
                    if taskB:
                        loss = self.model.loss(y_pre, labels, self.method)
                    else:
                        loss = self.model.loss(y_pre, labels)
            self.model.optimize(loss, tape)
            with tf.name_scope('train_accuracy'):
                if taskB:
                    acc = self.model.accuracyB(y_pre, labels)
                else:
                    acc = self.model.accuracyA(y_pre, labels)
        return y_pre, loss, acc

    @tf.function
    def _test_body(self, images, labels, taskB=False):
        with tf.device(self.device):
            with tf.name_scope('test_logits'):
                y_pre = self.model(images, trainable=False)
            with tf.name_scope('test_loss'):
                loss = self.model.loss(y_pre, labels)
            with tf.name_scope('test_accuracy'):
                if taskB:
                    acc = self.model.accuracyB(y_pre, labels)
                else:
                    acc = self.model.accuracyA(y_pre, labels)
        return y_pre, loss, acc

    def epoch_end(self, metrics, other=None):
        self.model.reset_accuracy()
        learning_rate = self.model.optimizer.lr(metrics['epoch']).numpy() if type(self.model.optimizer.lr) is tf.optimizers.schedules.ExponentialDecay else self.model.optimizer.lr
        tf.summary.experimental.set_step(metrics['epoch'])
        tf.summary.scalar('detail/epoch', metrics['epoch'])
        tf.summary.scalar('detail/time_per_step', metrics['time/epoch'])
        tf.summary.scalar('detail/learning_rate', learning_rate)
        tf.summary.scalar('train/loss', metrics['train_loss'])
        tf.summary.scalar('train/accuracy', metrics['train_accuracy'])
        tf.summary.scalar('test/lossA', metrics['test_lossA'])
        tf.summary.scalar('test/accuracyA', metrics['test_accuracyA'])
        if 'test_lossB' in metrics:
            tf.summary.scalar('test/accuracyB', metrics['test_accuracyB'])
            tf.summary.scalar('test/lossB', metrics['test_lossB'])
        if 'train_image' in other and len(other['train_image'].shape) == 4:
            tf.summary.image('train/image', other['train_image'])
            tf.summary.image('test/image', other['test_image'])

        if 'test_lossB' in metrics:
            print("epoch: %d  train_loss: %.4f  train_accuracy: %.3f test_lossA: %.4f  test_accuracyA: %.3f  test_lossB: %.4f  test_accuracyB: %.3f  time/epoch: %0.3fs" 
                                %(metrics['epoch'], metrics['train_loss'], metrics['train_accuracy'], metrics['test_lossA'], metrics['test_accuracyA'], metrics['test_lossB'], metrics['test_accuracyB'], metrics['time/epoch']))
        else:
            print("epoch: %d  train_loss: %.4f  train_accuracy: %.3f test_lossA: %.4f  test_accuracyA: %.3f  time/epoch: %0.3fs" 
                                %(metrics['epoch'], metrics['train_loss'], metrics['train_accuracy'], metrics['test_lossA'], metrics['test_accuracyA'], metrics['time/epoch']))
        self.util.write_log(message=metrics)
        if metrics['epoch'] % self.save_checkpoint_steps == 0:
            self.util.save_model(global_step=metrics['epoch'])
        return

    def train(self):
        board_writer = self.begin_train()

        if self.restore_dir is not None:
            self.util.restore_agent(self.model ,self.restore_dir)
        
        train_datasetA, valid_datasetA, test_datasetA = self.load()
        train_datasetB, valid_datasetB, test_datasetB = self.load(perm=True)

        # Graph for tensorboard
        tf.summary.trace_on(graph=True, profiler=True)
        with board_writer.as_default():
            Accuracy_A, Accuracy_B = [], []

            for i in range(1, self.n_epoch*2+1):
                if self.method == "EWC" and i == self.n_epoch + 1:
                    self.model.fissher_info(valid_datasetA, num_batches=1000)
                if i < self.n_epoch + 1:
                    train_dataset = train_datasetA
                    taskB_flag = False
                else:
                    train_dataset = train_datasetB
                    taskB_flag = True
                    self.model.star()
                
                start_time = time.time()
                for (train_images, train_labels) in train_dataset:
                    _, train_loss, train_accuracy = self._train_body(train_images, train_labels, taskB=taskB_flag)
                time_per_episode = time.time() - start_time

                for (test_images, test_labels) in test_datasetA:
                    _, test_lossA, test_accuracyA = self._test_body(test_images, test_labels)

                for (test_images, test_labels) in test_datasetB:
                    _, test_lossB, test_accuracyB = self._test_body(test_images, test_labels, taskB=True)

                # Training results
                metrics = OrderedDict({
                    "epoch": i,
                    "train_loss": train_loss.numpy(),
                    "train_accuracy":train_accuracy.numpy(),
                    "test_lossA": test_lossA.numpy(),
                    "test_accuracyA" : test_accuracyA.numpy(),
                    "test_lossB": test_lossB.numpy(),
                    "test_accuracyB" : test_accuracyB.numpy(),
                    "time/epoch": time_per_episode
                })
                other_metrics = OrderedDict({
                    "train_image" : train_images[:3],
                    "test_image" : test_images[:3]
                })

                Accuracy_A.append(test_accuracyA.numpy())
                Accuracy_B.append(test_accuracyB.numpy())
                self.epoch_end(metrics, other_metrics)
            self.progress_graph(Accuracy_A, Accuracy_B)
        
        return

    def progress_graph(self, taskA, taskB):
        plt.clf()
        n_epoch = np.arange(len(taskA))
        # プロット
        plt.plot(n_epoch, taskA, label="taskA")
        plt.plot(n_epoch, taskB, label="taskB")

        # 凡例の表示
        plt.legend()
        plt.grid()

        plt.savefig("./progress_{}.png".format(self.method))
        plt.close()
        return