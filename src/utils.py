# -*- coding: utf-8 -*-
import os, re
import numpy as np
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from tensorflow.python.client import device_lib


class Utils():
    def __init__(self, sess=None, prefix=None):
        dt_now = datetime.datetime.now()
        self.sess = sess
        self.res_dir = "results/"+dt_now.strftime("%y%m%d_%H%M%S")
        if prefix is not None:
            self.res_dir = self.res_dir + "_{}".format(prefix)
        self.log_dir = self.res_dir + "/log"
        self.tf_board = self.res_dir + "/tf_board"
        self.model_path = self.res_dir + "/model"
        self.saved_model_path = self.model_path + "/saved_model"

    def conf_log(self):
        if tf.io.gfile.exists(self.res_dir):
            tf.io.gfile.remove(self.res_dir)
        tf.io.gfile.makedirs(self.res_dir)
        return

    def initial(self):
        self.conf_log()
        if not os.path.isdir(self.log_dir):
            tf.io.gfile.makedirs(self.log_dir)
        return

    def write_configuration(self, message, _print=False):
        """
        設定をテキストに出力する
        parameters
        -------
        message : dict
        _print : True / False : terminalに表示するか
        """
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write("------Learning Details------\n")
            if _print:
                print("------Learning Details------")
            for key, info in message.items():
                f.write("%s : %s\n"%(key, info))
                if _print:
                    print("%s : %s"%(key, info))
            f.write("----------------------------\n")
            print("----------------------------")
        return 


    def write_log(self, message, test=False):
        """
        学習状況をテキストに出力する
        parameters
        -------
        message : dict
        test : bool
        """
        stats = []
        for key, info in message.items():
            stats.append("%s = %s" % (key, info))
        info = "%s\n"%(", ".join(stats))
        if test:
            with open(self.log_dir + '/test_log.txt', 'a') as f:
                f.write(str(info))
        else:
            with open(self.log_dir + '/log.txt', 'a') as f:
                f.write(str(info))
        return 

    def save_init(self, model, keep=5, n_hour=1):
        self.checkpoint = tf.train.Checkpoint(policy=model)
        self.saver =  tf.train.CheckpointManager(self.checkpoint,
                                                 directory=self.model_path,
                                                 keep_checkpoint_every_n_hours=n_hour,
                                                 max_to_keep=keep)
        return

    def save_model(self, global_step=None):
        if self.sess is not None:
            self.saver.save(self.sess, self.log_dir + "/model.ckpt"%global_step)
        else:
            self.saver.save(checkpoint_number=global_step)

    def restore_agent(self, model, log_dir=None):
        self.checkpoint = tf.train.Checkpoint(policy=model)
        self.checkpoint.restore(tf.train.latest_checkpoint(log_dir))
        return


    def restore_model(self, log_dir=None):
        assert log_dir is not None, 'Please set log_dir to restore checkpoint'
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver = tf.train.import_meta_graph("{}.meta".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Restore : {}.meta".format(ckpt.model_checkpoint_path))
            return True
        else:
            print ('Not Restore model in "{}"'.format(log_dir))
            return False

    def saved_model(self, x, y):
        '''
        x : Placeholder input
        y : Placeholder label or correct data
        '''
        builder = tf.saved_model.builder.SavedModelBuilder(self.model_path)
        signature = tf.saved_model.predict_signature_def(inputs={'inputs':x}, outputs={'label':y})
        builder.add_meta_graph_and_variables(sess=self.sess,
                                             tags=[tf.saved_model.tag_constants.SERVING],
                                             signature_def_map={'predict': signature})
        builder.save()
        return

    def construct_figure(self, x_test, decoded_imgs, step, n=10):
        '''
        元の画像と生成した画像10枚ずつを保存する
        parameters
        ----------
        x_test : input test image
        decoded_imgs : generate image
        returns
        -------
        '''
        plt.figure(figsize=(20, 4))
        for i in range(n):
            #  display original
            ax = plt.subplot(2, n, i + 1)
            try:
                ax.imshow(x_test[i].reshape(28, 28))
            except:
                ax.imshow(x_test[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(2, n, i + 1 + n)
            try:
                ax.imshow(decoded_imgs[i].reshape(28, 28))
            except:
                ax.imshow(decoded_imgs[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            plt.tight_layout()
        plt.savefig(self.log_dir + '/construct_figure_{}.png'.format(step))
        plt.close()

    def reconstruct_image(self, decoded_imgs, step):
        """
        VAEで出力した画像の図を作成する
        """
        amount_image = int(np.sqrt(decoded_imgs.shape[0]))
        fig = plt.figure(figsize=(amount_image, amount_image))
        gs = gridspec.GridSpec(amount_image, amount_image)
        gs.update(wspace=0.05, hspace=0.05)

        for i in range(amount_image**2):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            try:
                plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='Greys_r')
            except:
                plt.imshow(np.clip(decoded_imgs[i],0,255))

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.savefig(self.log_dir + '/construct_figure_{}.png'.format(step))

        plt.close(fig)

        return 

    def gan_plot(self, samples):
        """
        GANで生成した画像の図を作成する
        """
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(6, 6)
        gs.update(wspace=0.05, hspace=0.05)

        for i in range(36):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            try:
                plt.imshow(samples[i].reshape(28, 28), cmap='Greys_r')
            except:
                plt.imshow(np.clip(samples[i],0,255))

        i = 0
        while(True):
            name = self.log_dir + '/{}.png'.format(str(i).zfill(3))
            if os.path.isfile(name):
                i += 1
            else:
                plt.savefig(name, bbox_inches='tight')
                break

        plt.close(fig)

        return 
    
    def plot_figure(self, samples, iteration):
        fig = plt.figure(figsize=(6, 6))
        gs = gridspec.GridSpec(6, 6)
        gs.update(wspace=0.05, hspace=0.05)

        for i in range(36):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            try:
                plt.imshow(samples[i].reshape(28, 28), cmap='Greys_r')
            except:
                plt.imshow(np.clip(samples[i],0,255))

        name = self.log_dir + '/{}.png'.format(str(iteration).zfill(3))
        plt.savefig(name, bbox_inches='tight')
        plt.close(fig)

        return

    def plot_latent_space(self, z, label, step, n_label=10, xlim=None, ylim=None):
        """
        潜在空間zの可視化を行う
        """
        # onehot -> index
        label = np.argmax(label, axis=1)
        plt.clf()
        _, ax = plt.subplots(ncols=1, figsize=(8,8))
        color = cm.rainbow(np.linspace(0, 1, n_label))
        for l, c in zip(range(10), color):
            ix = np.where(label==l)[0]
            x = z[ix,0]
            y = z[ix, 1]
            c = np.tile(c, (x.shape[0],1))
            ax.scatter(x, y, c=c, label=l, s=8, linewidth=0)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        plt.savefig(self.log_dir + '/latent_space{}.png'.format(step))
        plt.close()
        return
