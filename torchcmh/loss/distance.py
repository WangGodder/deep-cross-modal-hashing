# -*- coding: utf-8 -*-
# @Time    : 2019/7/25
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
__all__ = ['focal_loss', 'hamming_dist', 'euclidean_dist']


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


def hamming_dist(hash1, hash2):
    """
    calculate hamming distance
    :param hash1: hash with value {-1, 1}, with shape (hash number, hash bit)
    :param hash2: hash with value {-1, 1} and same length of hash bit with hash1
    :return:
    """
    q = hash1.shape[-1]
    distH = 0.5 * (q - hash1.mm(hash2.transpose(0, 1)))
    return distH


def euclidean_dist(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    calculate euclidean distance as inner product
    :param tensor1:
    :param tensor2:
    :return:
    """
    dim1 = tensor1.shape[0]
    dim2 = tensor2.shape[0]
    multi = torch.matmul(tensor1, tensor2.t())
    a2 = torch.sum(torch.pow(tensor1, 2), dim=1, keepdim=True).expand(dim1, dim2)
    b2 = torch.sum(torch.pow(tensor2, 2), dim=1, keepdim=True).t().expand(dim1, dim2)
    dist = torch.sqrt(a2 + b2 - 2 * multi)
    return dist

