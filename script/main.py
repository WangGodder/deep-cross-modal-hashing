# -*- coding: utf-8 -*-
# @Time    : 2019/7/13
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torchcmh.run import run
import torch

torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    run()

