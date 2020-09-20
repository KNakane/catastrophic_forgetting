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
    args.gpu = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"

    # dataset
    dataset = Dataset(name='mnist')

    # model
    model = DNN(name='DNN',
                input_shape=dataset.input_shape,
                out_dim=dataset.output_dim,
                opt=args.opt,
                lr=args.lr)

    message = OrderedDict({
        "epoch":args.n_epoch,
        "batch_size": args.batch_size,
        "Optimizer":args.opt,
        "learning_rate":args.lr,
        "l2_norm": args.l2_norm,
        "GPU/CPU": args.gpu})

    # Training
    #trainer = Trainer(dataset=dataset, model=model, epoch=args.n_epoch, batch_size=args.batch_size, device=device)
    trainer = Trainer(FLAGS=args, message=message, data=dataset, model=model, name='DNN')
    trainer.train()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epoch', default=30, type=int, help='Input max epoch')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size')
    parser.add_argument('--opt', default='SGD', type=str, choices=['SGD','Momentum','Adadelta','Adagrad','Adam','RMSprop'])
    parser.add_argument('--lr', default=0.1, type=float, help='Input learning rate')
    parser.add_argument('--l2_norm', action='store_true', help='L2 normalization or not')
    parser.add_argument('--init_model', default=None, type=str, help='Choice the checkpoint directpry(ex. ./results/181225_193106/model)')
    parser.add_argument('--checkpoints_to_keep', default=5, type=int, help='checkpoint keep count')
    parser.add_argument('--keep_checkpoint_every_n_hours', default=1, type=int, help='checkpoint create hour')
    parser.add_argument('--save_checkpoint_steps', default=100, type=int, help='save checkpoint step')
    args = parser.parse_args()
    main(args)