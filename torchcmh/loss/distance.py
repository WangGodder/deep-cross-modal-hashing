# -*- coding: utf-8 -*-
# @Time    : 2019/7/25
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
__all__ = ['hamming_dist', 'euclidean_dist_matrix', 'euclidean_dist']


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


def euclidean_dist_matrix(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    calculate euclidean distance as inner product
    :param tensor1: a tensor with shape (a, c)
    :param tensor2: a tensor with shape (b, c)
    :return: the euclidean distance matrix which each point is the distance between a row in tensor1 and a row in tensor2.
    """
    dim1 = tensor1.shape[0]
    dim2 = tensor2.shape[0]
    multi = torch.matmul(tensor1, tensor2.t())
    a2 = torch.sum(torch.pow(tensor1, 2), dim=1, keepdim=True).expand(dim1, dim2)
    b2 = torch.sum(torch.pow(tensor2, 2), dim=1, keepdim=True).t().expand(dim1, dim2)
    dist = torch.sqrt(a2 + b2 - 2 * multi)
    return dist


def euclidean_dist(tensor1:torch.Tensor, tensor2:torch.Tensor):
    """
    calculate euclidean distance between two list of vector.
    :param tensor1: tensor with shape (a, b)
    :param tensor2: tensor with shape (a, b)
    :return:
    """
    sub = tensor1 - tensor2
    dist = torch.sqrt(torch.sum(torch.pow(sub, 2), dim=1))
    return dist