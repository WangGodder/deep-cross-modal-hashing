# coding: utf-8
# author: Godder
# Date: 2021/5/11
# Github: https://github.com/WangGodder

import sys
from torchcmh.create_trainer import create_new_algorithm

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            create_new_algorithm(sys.argv[i])
    else:
        create_new_algorithm("MyAlgorithm")
