# -*- coding: utf-8 -*-
# @Time    : 2019/7/25
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import datetime
from torchcmh.utils.logging import Logger
import os
import sys
import importlib
from torchcmh.utils.config_loader import Config

__all__ = ['run']


def run(config_path='default_config.yml', **kwargs):
    config = Config(config_path)
    method = config.training['method']
    data_name = config.training['dataName']
    img_dir = config.get_img_dir()
    bit = int(config.training['bit'])
    batch_size = int(config.training['batchSize'])
    device = config.training['device']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    cuda = bool(config.training['cuda'])
    kwargs['img_train_transform'] = config.img_training_transform
    kwargs['img_valid_transform'] = config.img_valid_transform
    kwargs['txt_train_transform'] = config.txt_training_transform
    kwargs['txt_valid_transform'] = config.txt_valid_transform
    # img_dir, img_list, tag_list, label_list = config.get_dataset_path(dataset_name)
    t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    sys.stdout = Logger(os.path.join('..', 'logs', method.upper(), data_name.upper(), t + '.txt'))
    if cuda:
        print("using gpu device: %s" % str(device))
    else:
        print("using cpu")
    print("training transform:")
    print("img:", config.img_training_transform)
    print("txt:", config.txt_training_transform)
    print("valid transform")
    print("img:", config.img_valid_transform)
    print("txt:", config.txt_valid_transform)
    get_train(method, data_name.upper(), img_dir, bit, batch_size=batch_size, cuda=cuda, **kwargs)


def get_train(method_name: str, dataset_name: str, img_dir: str, bit: int, **kwargs):
    package = "torchcmh.training"
    module = importlib.import_module('.' + method_name, package)
    train = getattr(module, 'train')
    t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    sys.stdout = Logger(os.path.join('..', 'logs', method_name, dataset_name.upper(), t + '.txt'))
    train(dataset_name, img_dir, bit, **kwargs)
