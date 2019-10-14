# -*- coding: utf-8 -*-
# @Time    : 2019/7/22
# @Author  : Godder
# @Github  : https://github.com/WangGodder


def change_learning_rate(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr


def adjust_learning_rate(optimizer, base_lr, epoch, stepsize=20, gamma=0.1,
                         linear_decay=False, final_lr=0, max_epoch=100):
    """Adjusts learning rate.

    Deprecated.
    """
    if linear_decay:
        # linearly decay learning rate from base_lr to final_lr
        frac_done = epoch / max_epoch
        lr = frac_done * final_lr + (1. - frac_done) * base_lr
    else:
        # decay learning rate by gamma for every stepsize
        lr = base_lr * (gamma ** (epoch // stepsize))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr