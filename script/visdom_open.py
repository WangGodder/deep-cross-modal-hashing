# coding: utf-8
# author: Godder
# Date: 2021/5/11
# Github: https://github.com/WangGodder
import os
import sys

if __name__ == '__main__':
    port = 8097
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    result = os.system("python -m visdom.server -port %i" % port)