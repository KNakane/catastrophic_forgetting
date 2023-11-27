import os, sys
import time
import copy
import numpy as np
import tensorflow as tf
from src.utils import Utils
from collections import OrderedDict
import matplotlib.pyplot as plt
from cpprb import ReplayBuffer


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
        self.embedding_dim = args.emb_dim
        self._mem_size = 25
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
    def _train_body(self, images, labels, continual=0):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                with tf.name_scope('train_logits'):
                    y_pre = self.model(images, trainable=True)
                with tf.name_scope('train_loss'):
                    if self.name == "ReplayThroughFeedback":
                        loss = self.model.loss(y_pre, images, labels)
                    else:
                        loss = self.model.loss(y_pre, labels, tape, self.method if continual > 0 else None)
                self.train_loss(loss)
            self.model.optimize(loss, tape)#, other_variables=tokens[-1])
            with tf.name_scope('train_accuracy'):
                self.acc_func(y_true=labels, y_pred=y_pre)
        return

    @tf.function
    def _train_si_body(self, images, labels, continual=0):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                with tf.name_scope('train_logits'):
                    y_pre = self.model(images, trainable=True)
                with tf.name_scope('train_loss'):
                    loss = self.model.loss(y_pre, labels)

                with tf.name_scope('total_loss'):
                     total_loss = loss + self.model.synaptic_intelligence()

        self.train_loss(total_loss)

        with tf.name_scope('gradient'):
            grads = tape.gradient(loss, self.model.trainable_variables)
        with tf.name_scope('reg_gradient'):
            reg_grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.model.optimize(total_loss, tape)

        with tf.name_scope('train_accuracy'):
            self.acc_func(y_true=labels, y_pred=y_pre)
        return grads, reg_grads

    @tf.function
    def _train_agem_body(self, images, labels, episodic_images, episodic_lables, continual=0):
        # https://www.slideshare.net/YuMaruyama/efficient-lifelong-learning-with-agem-iclr-2019-in-20190602-148340566
        # https://github.com/facebookresearch/agem/blob/master/fc_permute_mnist.py
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                with tf.name_scope('train_logits'):
                    y_pre = self.model(images, trainable=True)
                    if continual:
                        y_ref = self.model(episodic_images, trainable=True)
                with tf.name_scope('train_loss'):
                    loss = self.model.loss(y_pre, labels)
                    if continual:
                        loss_ref = self.model.loss(y_ref, episodic_lables)

        if continual:
            grads = tf.gradients(loss, self.model.trainable_variables)
            grads_ref = tf.gradients(loss_ref, self.model.trainable_variables)
            grads = self.model.agem_loss(grads, grads_ref)

            self.model.agem_optimize(grads)
        else:
            self.model.optimize(loss, tape)

        self.train_loss(loss)

        with tf.name_scope('train_accuracy'):
            self.acc_func(y_true=labels, y_pred=y_pre)
        return

    @tf.function
    def _train_er_body(self, images, labels, episodic_images, episodic_lables, continual=0):
        if continual:
            train_imgs = tf.concat([images, episodic_images], axis=0)
            train_labels = tf.concat([labels, episodic_lables], axis=0)
        else:
            train_imgs, train_labels = images, labels

        with tf.device(self.device):
            with tf.GradientTape() as tape:
                with tf.name_scope('train_logits'):
                    y_pre = self.model(train_imgs, trainable=True)

                with tf.name_scope('train_loss'):
                    loss = self.model.loss(y_pre, train_labels)

            self.train_loss(loss)

            self.model.optimize(loss, tape)

            with tf.name_scope('train_accuracy'):
                y_pre = self.model(images, trainable=True)
                self.acc_func(y_true=labels, y_pred=y_pre)
        return

    @tf.function
    def _train_lwf_body(self, images, labels, prev_model=None, continual=0):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                with tf.name_scope('train_logits'):
                    y_pre = self.model(images, trainable=True)
                    y_pres = tf.split(value=y_pre, num_or_size_splits=self.task_num, axis=1)
                    y_pre = y_pres.pop(continual)
                with tf.name_scope('train_loss'):
                    loss = self.model.loss(y_pre, labels)

                if prev_model is not None:
                    with tf.name_scope('old_logits'):
                        target_logits = tf.split(value=prev_model(images), num_or_size_splits=self.task_num, axis=1)
                        loss += self.model.lwf_loss(y_pres, target_logits)
                self.train_loss(loss)
            self.model.optimize(loss, tape)
            with tf.name_scope('train_accuracy'):
                self.acc_func(y_true=labels, y_pred=y_pre)
        return

    @tf.function
    def _train_hnet_body(self, images, labels, tokens=None, continual=0):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                with tf.name_scope('train_logits'):
                    y_pre = self.model(images, tokens[-1], trainable=True)
                with tf.name_scope('train_loss'):
                    weights = [self.model.hnet(token) for token in tokens]
                    loss = self.model.loss(y_pre, labels, tape, self.method, weights)
                    self.train_loss(loss)
            self.model.optimize(loss, tape)#, other_variables=tokens[-1])
            with tf.name_scope('train_accuracy'):
                self.acc_func(y_true=labels, y_pred=y_pre)

        return

    @tf.function
    def _test_body(self, images, labels, task_num=0, token=None):
        with tf.device(self.device):
            with tf.name_scope('test_logits'):
                if self.method == "HyperNet":
                    y_pre = self.model(images, token, trainable=False)
                elif self.method == "LwF":
                    y_pre = self.model(images, trainable=False)
                    y_pres = tf.split(value=y_pre, num_or_size_splits=self.task_num, axis=1)
                    y_pre = y_pres.pop(task_num)
                else:
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
            self.util.restore_agent(self.model, self.restore_dir)

        # previous model for LwF(update in Line 259)
        prev_model = None

        # init OMEGA
        if self.method == "SI":
            self.model.star()
            self.model.omega_info()

        if self.method in ["A-GEM", "ER"]:
            self.episodic_memory = ReplayBuffer(
                self._mem_size * self.task_num * self.data.output_dim,
                env_dict={"imgs": {"shape": self.data.input_shape},
                          "labels": {"shape": self.data.output_dim}})

        # load dataset
        train_dataset, valid_dataset, test_dataset = self.load()
        train_datasets, valid_datasets, test_datasets = [train_dataset], [valid_dataset], [test_dataset]

        # task embedding
        task_embedding = []
        for i in range(self.task_num):
            if self.method == "HyperNet":
                embed = tf.random.normal([self.embedding_dim]) / 10
                embed = tf.Variable(embed, trainable=True)
                task_embedding.append(embed)
            else:
                task_embedding.append(None)


        for _ in range(self.task_num - 1):
            train_dataset, valid_dataset, test_dataset = self.load(perm=True)
            train_datasets.append(train_dataset)
            valid_datasets.append(valid_dataset)
            test_datasets.append(test_dataset)

        all_loss = np.zeros((self.n_epoch*self.task_num, self.task_num))
        all_accuracy = np.zeros((self.n_epoch*self.task_num, self.task_num))

        # Graph for tensorboard
        tf.summary.trace_on(graph=True, profiler=True)
        with board_writer.as_default():
            total_epoch = 1
            for n_task, (train_dataset, valid_dataset) in enumerate(zip(train_datasets, valid_datasets)):

                for epoch in range(self.n_epoch):
                    start_time = time.time()
                    for (train_images, train_labels) in train_dataset:
                        if self.method == "HyperNet":
                            self._train_hnet_body(train_images,
                                                  train_labels,
                                                  task_embedding[:n_task+1],
                                                  continual=n_task)
                        elif self.method == "SI":
                            grads, reg_grads = self._train_si_body(train_images,
                                                                   train_labels,
                                                                   continual=n_task)

                            for i, (grad_cross, grad_all) in enumerate(zip(grads, reg_grads)):
                                self.model.omega[i] += self.model.lr * grad_cross * grad_all


                        elif self.method == "LwF":
                            self._train_lwf_body(train_images,
                                                 train_labels,
                                                 prev_model=prev_model,
                                                 continual=n_task)

                        elif self.method in ["A-GEM", "ER"]:
                            if self.episodic_memory.get_stored_size():
                                epi_batch = 256 if self.episodic_memory.get_stored_size() > 256 else self.episodic_memory.get_stored_size()
                                sample = self.episodic_memory.sample(epi_batch)
                                episodic_img, episodic_label = sample["imgs"], sample["labels"]
                            else:
                                episodic_img, episodic_label = None, None

                            train_body = self._train_agem_body if self.method == "A-GEM" else self._train_er_body
                            train_body(train_images,
                                       train_labels,
                                       episodic_img,
                                       episodic_label,
                                       continual=n_task)

                        else:
                            self._train_body(train_images,
                                             train_labels,
                                             continual=n_task)
                    time_per_episode = time.time() - start_time
                    # trainiing metricsを記録
                    train_loss = self.train_loss.result().numpy()
                    train_accuracy = self.acc_func.result().numpy()
                    # 訓練履歴のリセット
                    self.train_loss.reset_states()
                    self.acc_func.reset_states()

                    test_losses, test_accuracy, average_accuracy = [], [], []
                    for test_task, test_dataset in enumerate(test_datasets):
                        for (test_images, test_labels) in test_dataset:
                            self._test_body(test_images, test_labels, test_task, task_embedding[test_task])
                        # test lossを記録
                        test_losses.append(self.test_loss.result().numpy())
                        test_accuracy.append(self.acc_func.result().numpy())
                        if n_task >= test_task:
                            average_accuracy.append(self.acc_func.result().numpy())
                        # 訓練履歴のリセット
                        self.test_loss.reset_states()
                        self.acc_func.reset_states()

                    all_loss[total_epoch-1] = test_losses
                    all_accuracy[total_epoch-1] = test_accuracy

                    # Training results
                    metrics = OrderedDict({
                        "epoch": total_epoch,
                        "train_loss": train_loss,
                        "train_accuracy":train_accuracy,
                        "test_loss": test_losses,
                        "test_accuracy" : test_accuracy,
                        "average_accuracy": sum(average_accuracy) / len(average_accuracy),
                        "time/epoch": time_per_episode
                    })
                    other_metrics = OrderedDict({
                        "train_image" : train_images[:3],
                        "test_image" : test_images[:3]
                    })

                    total_epoch += 1
                    self.epoch_end(metrics, other_metrics)

                if self.method is not None:
                    self.model.star()
                if self.method in ["EWC", "OnlineEWC"]:
                    self.model.fissher_info(valid_dataset, num_batches=1000, online=self.__online)
                if self.method in ["SI"]:
                    self.model.omega_info()
                if self.method in ["HyperNet"]:
                    weights = [self.model.hnet(token) for token in task_embedding[:n_task+1]]
                    self.model.set_weights_snapshots(weights)
                if self.method in ["A-GEM", "ER"]:
                    for k, (img, labels) in enumerate(train_dataset):
                        self.episodic_memory.add(imgs=img,
                                                 labels=labels)
                        if k > int(self._mem_size * self.data.output_dim / self.batch_size):
                            break
                if self.method in ["LwF"]:
                    prev_model = copy.deepcopy(self.model)
                    #self.model.add_layer(new_class_num=self.data.output_dim)
                if self.method in ["A-GEM"]:
                    self.model.omega_info()

            self.progress_graph(all_loss, all_accuracy)
        
        return

    def forgetting_measure(self, now_task_num, accuracy_list):
        base = accuracy_list[now_task_num]
        fj = max([accuracy_list[i] - base for i in range(now_task_num)])

        return


    def progress_graph(self, loss, accuracy):
        plt.clf()
        n_epoch = np.arange(loss.shape[0]) + 1
        fig = plt.figure()

        # lossに関するグラフ
        ax1 = fig.add_subplot(2, 1, 1)
        for i in range(loss.shape[1]):
            ax1.plot(n_epoch, loss[:,i], label="task{}".format(i+1))
        ax1.legend()
        ax1.set_ylabel('loss')
        ax1.grid()

        ax2 = fig.add_subplot(2, 1, 2)
        for i in range(accuracy.shape[1]):
            ax2.plot(n_epoch, accuracy[:,i], label="task{}".format(i+1))
        ax2.legend()
        ax2.set_ylabel('accuracy')
        ax2.grid()
        ax2.set_ylim(0, 1)

        plt.savefig("./progress_{}.png".format(self.method))
        plt.clf()
        plt.close()
        return