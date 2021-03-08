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
        self.task_num = args.task_num
        self.method = args.method
        self.__online = True if self.method == "OnlineEWC" else False
        self.n_epoch = args.n_epoch
        self.save_checkpoint_steps = self.n_epoch / 10 if args.save_checkpoint_steps is None else args.save_checkpoint_steps
        self.checkpoints_to_keep = args.checkpoints_to_keep
        self.keep_checkpoint_every_n_hours = args.keep_checkpoint_every_n_hours
        self.batch_size = args.batch_size
        self.restore_dir = args.init_model
        self.device = args.gpu
        self.util = Utils(prefix=self.name)
        self.util.initial()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.acc_func = tf.keras.metrics.CategoricalAccuracy()

    def load(self, perm=False):
        train, valid, test = self.data.get()
        (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = train, valid, test
        if perm:
            img_list = [x_train[5]]
            x_train, x_valid, x_test = self.data.permutation(x_train, x_valid, x_test)
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
    def _train_body(self, images, labels, continual=False):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                with tf.name_scope('train_logits'):
                    y_pre = self.model(images)
                with tf.name_scope('train_loss'):
                    if self.name == "ReplayThroughFeedback":
                        loss = self.model.loss(y_pre, images, labels)
                    else:
                        loss = self.model.loss(y_pre, labels, self.method if continual else None)
                    self.train_loss(loss)
            self.model.optimize(loss, tape)
            with tf.name_scope('train_accuracy'):
                self.acc_func(y_true=labels, y_pred=y_pre)
        return

    @tf.function
    def _test_body(self, images, labels, taskB=False):
        with tf.device(self.device):
            with tf.name_scope('test_logits'):
                y_pre = self.model(images, trainable=False)
            with tf.name_scope('test_loss'):
                if self.name == "ReplayThroughFeedback":
                    loss = self.model.loss(y_pre, images, labels)
                else:
                    loss = self.model.loss(y_pre, labels)
                self.test_loss(loss)
            with tf.name_scope('test_accuracy'):
                self.acc_func(y_true=labels, y_pred=y_pre)
        return

    def epoch_end(self, metrics, other=None):
        print_format = "epoch: %d  train_loss: %.4f  train_acc: %.3f  "%(metrics['epoch'], metrics['train_loss'], metrics['train_accuracy'])
        learning_rate = self.model.optimizer.lr(metrics['epoch']).numpy() if type(self.model.optimizer.lr) is tf.optimizers.schedules.ExponentialDecay else self.model.optimizer.lr
        tf.summary.experimental.set_step(metrics['epoch'])
        tf.summary.scalar('detail/epoch', metrics['epoch'])
        tf.summary.scalar('detail/time_per_step', metrics['time/epoch'])
        tf.summary.scalar('detail/learning_rate', learning_rate)
        tf.summary.scalar('train/loss', metrics['train_loss'])
        tf.summary.scalar('train/accuracy', metrics['train_accuracy'])
        for i, (loss, acc) in enumerate(zip(metrics['test_loss'], metrics['test_accuracy'])):
            tf.summary.scalar('test/loss{}'.format(i+1), loss)
            tf.summary.scalar('test/accuracy{}'.format(i+1), acc)
            print_format += "test_loss{0:1d}: {1:.4f}  test_acc{0:1d}: {2:.3f}  ".format(i, loss, acc)
        if 'train_image' in other and len(other['train_image'].shape) == 4:
            tf.summary.image('train/image', other['train_image'])
            tf.summary.image('test/image', other['test_image'])

        print_format += "time/epoch: %0.3fs"%metrics['time/epoch']
        print(print_format)
    
        self.util.write_log(message=metrics)
        if metrics['epoch'] % self.save_checkpoint_steps == 0:
            self.util.save_model(global_step=metrics['epoch'])
        return

    def train(self):
        board_writer = self.begin_train()

        if self.restore_dir is not None:
            self.util.restore_agent(self.model ,self.restore_dir)
        
        train_dataset, valid_dataset, test_dataset = self.load()
        train_datasets, valid_datasets, test_datasets = [train_dataset], [valid_dataset], [test_dataset]
        for _ in range(self.task_num - 1):
            train_dataset, valid_dataset, test_dataset = self.load(perm=True)
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            test_datasets.append(test_dataset)


        # Graph for tensorboard
        tf.summary.trace_on(graph=True, profiler=True)
        with board_writer.as_default():
            total_epoch = 1
            for n_task, (train_dataset, valid_dataset) in enumerate(zip(train_datasets, valid_datasets)):
                for epoch in range(self.n_epoch):
                
                    start_time = time.time()
                    for (train_images, train_labels) in train_dataset:
                        self._train_body(train_images, train_labels, continual=bool(n_task))
                    time_per_episode = time.time() - start_time
                    # trainiing metricsを記録
                    train_loss = self.train_loss.result().numpy()
                    train_accuracy = self.acc_func.result().numpy()
                    # 訓練履歴のリセット
                    self.train_loss.reset_states()
                    self.acc_func.reset_states()

                    if self.method in ["EWC", "OnlineEWC"] and epoch == self.n_epoch-1:
                        self.model.fissher_info(valid_dataset, num_batches=1000, online=self.__online)
                        self.model.star()

                    test_losses, test_accuracy = [], []
                    for test_dataset in test_datasets:
                        for (test_images, test_labels) in test_dataset:
                            self._test_body(test_images, test_labels)
                        # test lossを記録
                        test_losses.append(self.test_loss.result().numpy())
                        test_accuracy.append(self.acc_func.result().numpy())
                        # 訓練履歴のリセット
                        self.test_loss.reset_states()
                        self.acc_func.reset_states()

                    # Training results
                    metrics = OrderedDict({
                        "epoch": total_epoch,
                        "train_loss": train_loss,
                        "train_accuracy":train_accuracy,
                        "test_loss": test_losses,
                        "test_accuracy" : test_accuracy,
                        "time/epoch": time_per_episode
                    })
                    other_metrics = OrderedDict({
                        "train_image" : train_images[:3],
                        "test_image" : test_images[:3]
                    })

                    #Accuracy_A.append(test_accuracyA.numpy())
                    #Accuracy_B.append(test_accuracyB.numpy())
                    total_epoch += 1
                    self.epoch_end(metrics, other_metrics)
            #self.progress_graph(Accuracy_A, Accuracy_B)
        
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