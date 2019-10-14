# -*- coding: utf-8 -*-
# @Time    : 2019/7/22
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch


def l2_norm(input):
    """Perform l2 normalization operation on an input vector.
    code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
    """
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output
