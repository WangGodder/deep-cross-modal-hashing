# -*- coding: utf-8 -*-
# @Time    : 2019/10/28
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
__all__ = ['focal_loss', 'cosine', 'vector_length']


def focal_loss(logit: torch.Tensor, gamma, alpha=1, eps=1e-5):
    """
    focal loss: /alpha * (1 - logit)^{gamma} * log^{logit}
    :param logit: logit value 0 ~ 1
    :param gamma:
    :param alpha:
    :param eps: a tiny value prevent log(0)
    :return:
    """
    return alpha * -torch.pow(1 - logit, gamma) * torch.log(logit + eps)


def cosine(hash1: torch.Tensor, hash2: torch.Tensor):
    inter = torch.matmul(hash1, hash2.t())
    length1 = vector_length(hash1, keepdim=True)
    length2 = vector_length(hash2, keepdim=True)
    return torch.div(inter, torch.matmul(length1, length2.t()))


def vector_length(vector: torch.Tensor, keepdim=False):
    if len(vector.shape) > 2:
        vector = vector.unsqueeze(0)
    return torch.sqrt(torch.sum(torch.pow(vector, 2), dim=-1, keepdim=keepdim))
