import os, sys, re
import time
import numpy as np
import tensorflow as tf
from src.utils import Utils
from collections import OrderedDict
import matplotlib.pyplot as plt
from src.hooks import SavedModelBuilderHook, MyLoggerHook


class Trainer():
    def __init__(self,
                 FLAGS,
                 message,
                 datasets,
                 model,
                 name):
        self.checkpoints_to_keep = FLAGS.checkpoints_to_keep
        self.keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
        self.max_steps = FLAGS.n_epoch
        self.save_checkpoint_steps = self.max_steps / 10 if FLAGS.save_checkpoint_steps is None else FLAGS.save_checkpoint_steps
        self.batch_size = FLAGS.batch_size
        self.name = name
        self.message = message
        self.data = datasets[0]
        self.datasets = datasets
        self.global_step = tf.train.get_or_create_global_step()
        self.model = model
        self.method = FLAGS.method
        self.restore_dir = FLAGS.init_model
        self.device = FLAGS.gpu
        self.util = Utils(prefix=self.name)
        self.util.initial()

    def load(self, perm=False):
        train_inputs = tf.placeholder(tf.float32, [None, self.data.size, self.data.size, self.data.channel], name='train_inputs')
        train_labels = tf.placeholder(tf.float32, shape=[None, 10], name='train_labels')

        valid_inputs = tf.placeholder(tf.float32, [None, self.data.size, self.data.size, self.data.channel], name='valid_inputs')
        valid_labels = tf.placeholder(tf.float32, shape=[None, 10], name='valid_labels')

        test_inputs = tf.placeholder(tf.float32, [None, self.data.size, self.data.size, self.data.channel], name='test_inputs')
        test_labels = tf.placeholder(tf.float32, shape=[None, 10], name='test_labels')

        return (train_inputs, train_labels), (valid_inputs, valid_labels), (test_inputs, test_labels)

    def build_logits(self, train_data, train_ans, valid_data, valid_ans, test_data, test_ans):
        # train
        self.train_logits = self.model.inference(train_data)
        self.train_loss = self.model.loss(self.train_logits, train_ans)
        opt_op = self.model.optimize(self.train_loss, self.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([opt_op] + update_ops)

        self.train_loss_ = self.model.loss(self.train_logits, train_ans, mode=self.method)
        reopt_op = self.model.optimize(self.train_loss_, self.global_step)
        self.retrain_op = tf.group([reopt_op] + update_ops)

        self.train_accuracy = self.model.evaluate(self.train_logits, train_ans)

        # validation
        self.valid_logits = self.model.test_inference(valid_data, reuse=True)
        self.valid_loss = self.model.loss(self.valid_logits, valid_ans)
        self.valid_accuracy = self.model.evaluate(self.valid_logits, valid_ans)

        # test
        self.test_logits = self.model.test_inference(test_data, reuse=True)
        self.test_loss = self.model.loss(self.test_logits, test_ans)
        self.test_accuracy = self.model.evaluate(self.test_logits, test_ans)

        return

    def build_fissher_info(self, images, labels):
        logits = self.model.inference(images, reuse=True)
        ll = tf.nn.log_softmax(logits)
        self.grads = tf.gradients(ll, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
        return


    def hook_append(self, metrics, signature_def_map=None):
        """
        hooksをまとめる関数
        """
        hooks = []
        hooks.append(tf.train.NanTensorHook(self.train_loss))
        hooks.append(MyLoggerHook(self.message, self.util.log_dir, metrics, every_n_iter=100))
        return hooks

    def summary(self, train_inputs, valid_inputs, test_inputs):
        """
        tensorboardに表示するデータをまとめる関数
        """
        # tensorboard
        tf.summary.scalar('train/loss', self.train_loss)
        tf.summary.scalar('train/accuracy', self.train_accuracy)
        tf.summary.scalar('train/Learning_rate', self.model.optimizer.lr)
        tf.summary.scalar('valid/loss', self.valid_loss)
        tf.summary.scalar('valid/accuracy', self.valid_accuracy)
        tf.summary.scalar('test/loss', self.test_loss)
        tf.summary.scalar('test/accuracy', self.test_accuracy)
        if len(train_inputs.shape) == 4:
            tf.summary.image('train/image', train_inputs)
            tf.summary.image('valid/image', valid_inputs)
            tf.summary.image('test/image', test_inputs)
        merged = tf.summary.merge_all()
        return merged

    def train(self):
        self.util.initial()
        # train
        (train_inputs, train_labels), (valid_inputs, valid_labels), (test_inputs, test_labels) = self.load()
        self.build_logits(train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels)
        self.build_fissher_info(valid_inputs, valid_labels)

        merged = self.summary(train_inputs, valid_inputs, test_inputs)

        # Measured Time
        step_time = tf.placeholder(tf.float32, name='step_time')
        calc_time = tf.summary.scalar('train/time_per_epoch', step_time)
        board4time = tf.summary.merge([calc_time])

        saver = tf.train.Saver(
            max_to_keep=self.checkpoints_to_keep,
            keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours
        )

        with tf.Session() as session:
            writer = tf.summary.FileWriter(self.util.tf_board, session.graph)
            if self.restore_dir is not None:
                if not re.search(r'model', self.restore_dir):
                    self.restore_dir = self.restore_dir + 'model/'
                ckpt = tf.train.get_checkpoint_state(self.restore_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    self.saver.restore(session, ckpt.model_checkpoint_path)
                    print('Restore -> {}'.format(self.restore_dir))
            else:
                session.run(tf.global_variables_initializer())
                print('Not restore init model')


            # Training each task
            start = time.time()
            for n_task, data in enumerate(self.datasets):
                train_op = self.retrain_op if n_task else self.train_op
                for step in range(1, self.max_steps+1):
                    iteration_time = time.time()
                    _, x_train, y_train = data.next_batch(batch_size=self.batch_size)
                    _, train_loss, train_acc = session.run([train_op, self.train_loss, self.train_accuracy],
                                                           feed_dict={
                                                             train_inputs: x_train,
                                                             train_labels: y_train
                                                           })
                    iteration_end = time.time() - iteration_time
                    summary_time = session.run(board4time, feed_dict={step_time:iteration_end})
                    #writer.add_summary(summary, step)
                    writer.add_summary(summary_time, step)

                    if step == 1 or step % 100 == 0:
                        metrics = OrderedDict({
                            "global_step": step,
                            "train loss": '{:.3f}'.format(train_loss),
                            "train accuracy":'{:.3f}'.format(train_acc)
                        })

                        for i, test_data in enumerate(self.datasets):
                            _, x_test, y_test = test_data.next_batch(batch_size=10000, test=True)
                            test_loss, test_acc = session.run(
                                [self.test_loss, self.test_accuracy],
                                feed_dict={
                                    test_inputs: x_test,
                                    test_labels: y_test
                                }
                            )
                            metrics["test_loss{}".format(i+1)] = '{:.3f}'.format(test_loss)
                            metrics["test_acc{}".format(i+1)] = '{:.3f}'.format(test_acc)

                    
                        metrics["time_per_epoch"] = '{:.3f}'.format(iteration_end)

                        self.util.write_log(message=metrics, cout=True)
                        saver.save(session, self.util.model_path + '/model', global_step=step)

                # save model parameter
                valid_num = 1000
                _, x_valid, y_valid = data.next_batch(batch_size=valid_num, valid=True)
                gradients = session.run(self.grads, feed_dict={valid_inputs: x_valid, valid_labels: y_valid})
                self.model.fissher_info(gradients, valid_num)
                self.model.set_old_val()


            elapsed_time = time.time() - start
            print(elapsed_time)
            #self.progress_graph(Accuracy_A, Accuracy_B)
            writer.close()

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

        plt.savefig("./progress.png")
        plt.close()
        return