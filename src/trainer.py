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
                 data,
                 model,
                 name):
        self.checkpoints_to_keep = FLAGS.checkpoints_to_keep
        self.keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
        self.max_steps = FLAGS.n_epoch
        self.save_checkpoint_steps = self.max_steps / 10 if FLAGS.save_checkpoint_steps is None else FLAGS.save_checkpoint_steps
        self.batch_size = FLAGS.batch_size
        self.name = name
        self.message = message
        self.data = data
        self.global_step = tf.train.get_or_create_global_step()
        self.model = model
        self.restore_dir = FLAGS.init_model
        self.device = FLAGS.gpu
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
        valid_dataset = self.data.load(x_valid, y_valid, batch_size=200, is_training=False, is_valid=True)
        test_dataset = self.data.load(x_test, y_test, batch_size=self.batch_size*10, is_training=False)
        
        self.train_iter = train_dataset.make_initializable_iterator()
        self.valid_iter = valid_dataset.make_initializable_iterator()
        self.test_iter = test_dataset.make_initializable_iterator()
        train_inputs, train_labels = self.train_iter.get_next()
        valid_inputs, valid_labels = self.valid_iter.get_next()
        test_inputs, test_labels = self.test_iter.get_next()
        return (train_inputs, train_labels), (valid_inputs, valid_labels), (test_inputs, test_labels)

    def build_logits(self, train_data, train_ans, valid_data, valid_ans, test_data, test_ans):
        # train
        self.train_logits = self.model.inference(train_data)
        self.train_loss = self.model.loss(self.train_logits, train_ans)
        #self.predict = self.model.predict(train_data)
        opt_op = self.model.optimize(self.train_loss, self.global_step)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.train_op = tf.group([opt_op] + update_ops)
        self.train_accuracy = self.model.evaluate(self.train_logits, train_ans, prefix='A')

        # validation
        self.valid_logits = self.model.test_inference(valid_data, reuse=True)
        self.valid_loss = self.model.loss(self.valid_logits, valid_ans)
        self.valid_accuracy = self.model.evaluate(self.valid_logits, valid_ans, prefix='A')

        # test
        self.test_logits = self.model.test_inference(test_data, reuse=True)
        self.test_loss = self.model.loss(self.test_logits, test_ans)
        self.test_accuracy = self.model.evaluate(self.test_logits, test_ans, prefix='A')

        return

    def hook_append(self, metrics, signature_def_map=None):
        """
        hooksをまとめる関数
        """
        hooks = []
        hooks.append(tf.train.NanTensorHook(self.train_loss))
        hooks.append(MyLoggerHook(self.message, self.util.log_dir, metrics, every_n_iter=100))
        hooks.append(SavedModelBuilderHook(self.util.saved_model_path, signature_def_map))
        if self.max_steps:
            hooks.append(tf.train.StopAtStepHook(last_step=self.max_steps))
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
        return

    def before_train(self):

        def init_fn(scaffold, session):
            session.run([self.train_iter.initializer, self.valid_iter.initializer, self.test_iter.initializer],
                        feed_dict={self.data.features_placeholder: self.data.x_train,
                                   self.data.labels_placeholder: self.data.y_train,
                                   self.data.valid_placeholder: self.data.x_valid,
                                   self.data.valid_labels_placeholder: self.data.y_valid,
                                   self.data.test_placeholder: self.data.x_test,
                                   self.data.test_labels_placeholder: self.data.y_test})
        
        # create saver
        self.saver = tf.train.Saver(
                max_to_keep=self.checkpoints_to_keep,
                keep_checkpoint_every_n_hours=self.keep_checkpoint_every_n_hours)

        scaffold = tf.train.Scaffold(
            init_fn=init_fn,
            saver=self.saver)

        tf.logging.set_verbosity(tf.logging.INFO)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        # saved model
        signature_def_map = {
                        'predict': tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={'inputs': tf.saved_model.utils.build_tensor_info(self.data.features_placeholder)},
                            outputs={'predict': tf.saved_model.utils.build_tensor_info(self.test_logits)},
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,)
                        }

        metrics = OrderedDict({
            "global_step": self.global_step,
            "train loss": self.train_loss,
            "train accuracy":self.train_accuracy,
            "valid loss": self.valid_loss,
            "valid accuracy":self.valid_accuracy,
            "test loss": self.test_loss,
            "test accuracy":self.test_accuracy})

        hooks = self.hook_append(metrics, signature_def_map)

        session = tf.train.MonitoredTrainingSession(
            config=config,
            checkpoint_dir=self.util.model_path,
            hooks=hooks,
            scaffold=scaffold,
            save_summaries_steps=1,
            save_checkpoint_steps=self.save_checkpoint_steps,
            summary_dir=self.util.tf_board)
        
        return session


    def train(self):
        self.util.conf_log()
        (train_inputs, train_labels), (valid_inputs, valid_labels), (test_inputs, test_labels) = self.load()
        self.build_logits(train_inputs, train_labels, valid_inputs, valid_labels, test_inputs, test_labels)
        self.summary(train_inputs, valid_inputs, test_inputs)
        session = self.before_train()

        with session:
            if self.restore_dir is not None:
                if not re.search(r'model', self.restore_dir):
                    self.restore_dir = self.restore_dir + 'model/'
                ckpt = tf.train.get_checkpoint_state(self.restore_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    self.saver.restore(session, ckpt.model_checkpoint_path)
            while not session.should_stop():
                session.run([self.train_op])
        return 

    def progress_graph(self, taskA, taskB):
        plt.clf()
        n_epoch = list(range(self.max_steps))
        # プロット
        plt.plot(n_epoch, taskA, label="taskA")
        plt.plot(n_epoch, taskB, label="taskB")

        # 凡例の表示
        plt.legend()
        plt.grid()

        plt.savefig("./progress.png")
        plt.close()
        return