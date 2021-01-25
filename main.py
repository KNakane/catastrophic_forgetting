import os, sys, re
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
from src.model import CNN, DNN
from src.data import Dataset
from src.trainer import Trainer
from collections import OrderedDict

def find_gpu():
    device_list = device_lib.list_local_devices()
    for device in device_list:
        if re.match('/device:GPU', device.name):
            return 0
    return -1

def main(args):
    # GPU setting
    gpu = find_gpu()
    FLAGS.gpu = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    # dataset
    dataset = Dataset(name='mnist')

    # model
    model = DNN(name='DNN',
                input_shape=dataset.input_shape,
                out_dim=dataset.output_dim,
                opt=FLAGS.opt,
                lr=FLAGS.lr)

    message = OrderedDict({
        "epoch":FLAGS.n_epoch,
        "batch_size": FLAGS.batch_size,
        "Optimizer":FLAGS.opt,
        "learning_rate":FLAGS.lr,
        "l2_norm": FLAGS.l2_norm,
        "GPU/CPU": FLAGS.gpu})

    # Training
    #trainer = Trainer(dataset=dataset, model=model, epoch=FLAGS.n_epoch, batch_size=FLAGS.batch_size, device=device)
    trainer = Trainer(FLAGS=FLAGS, message=message, data=dataset, model=model, name='DNN')
    trainer.train()
    return

if __name__ == "__main__":
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('n_epoch', 1000, 'Input max epoch')
    flags.DEFINE_integer('batch_size', 32, 'Input batch size')
    flags.DEFINE_string('method', "EWC", "[EWC, L2]")
    flags.DEFINE_string('opt', 'SGD', "['SGD','Momentum','Adadelta','Adagrad','Adam','RMSprop']")
    flags.DEFINE_float('lr', 0.1, 'Input learning rate')
    flags.DEFINE_bool('l2_norm', False, 'L2 normalization or not')
    flags.DEFINE_string('init_model', None, 'Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    flags.DEFINE_integer('checkpoints_to_keep', 5, 'checkpoint keep count')
    flags.DEFINE_integer('keep_checkpoint_every_n_hours', 1, 'checkpoint create hour')
    flags.DEFINE_integer('save_checkpoint_steps', 100, 'save checkpoint step')
    flags.DEFINE_integer('gpu', -1, 'Using GPU')
    tf.app.run()