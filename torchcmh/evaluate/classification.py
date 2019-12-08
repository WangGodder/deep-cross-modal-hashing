# -*- coding: utf-8 -*-
# @Time    : 2019/12/2
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
__all__ = ['multi_label_acc']


def multi_label_acc(predict: torch.Tensor, label: torch.Tensor):
    """
    Multi-label acc evaluate.
    acc = \frac{predict right label}{number of label}
    :param predict: the predict label
    :param label: the ground truth label with same shape as predict
    :return: the mean accuracy of all predict instances.
    """
    assert predict.shape == label.shape
    label_num = torch.sum(label, dim=-1)
    acc = 0
    for i in range(predict.size(0)):
        _, predict_ind = torch.topk(predict[i, :], int(label_num[i]))
        right_num = torch.sum(label[i][predict_ind])
        acc += (right_num / label_num[i]).item()
    acc /= predict.size(0)
    return acc


