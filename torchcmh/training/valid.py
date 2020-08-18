# -*- coding: utf-8 -*-
# @Time    : 2019/5/17
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torch import nn
from ..utils import calc_map_k, calc_precisions_topn, calc_precisions_hash
from torch.utils.data import DataLoader
from ..dataset.base import CrossModalValidBase
from tqdm import tqdm
import torch
import numpy as np


def valid(img_model: nn.Module, txt_model: nn.Module, dataset: CrossModalValidBase, bit, batch_size, drop_integer=False, return_hash=False):
    # get query img and txt binary code
    dataset.query()
    qB_img = get_img_code(img_model, dataset, bit, batch_size, drop_integer)
    qB_txt = get_txt_code(txt_model, dataset, bit, batch_size, drop_integer)
    query_label = dataset.get_all_label()
    # get retrieval img and txt binary code
    dataset.retrieval()
    rB_img = get_img_code(img_model, dataset, bit, batch_size, drop_integer)
    rB_txt = get_txt_code(txt_model, dataset, bit, batch_size, drop_integer)
    retrieval_label = dataset.get_all_label()
    mAPi2t = calc_map_k(qB_img, rB_txt, query_label, retrieval_label)
    mAPt2i = calc_map_k(qB_txt, rB_img, query_label, retrieval_label)
    if return_hash:
        return mAPi2t, mAPt2i, qB_img.cpu(), qB_txt.cpu(), rB_img.cpu(), rB_txt.cpu()
    return mAPi2t, mAPt2i


def get_img_code(img_model: nn.Module, dataset: CrossModalValidBase, bit, batch_size, drop_integer=False, cuda=True):
    dataset.img_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=drop_integer)
    B_img = torch.zeros(len(dataset), bit, dtype=torch.float)
    if cuda:
        B_img = B_img.cuda()
    img_model.eval()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        img = data['img']  # type: torch.Tensor
        if cuda:
            img = img.cuda()

        f = img_model(img)
        B_img[index, :] = f.data
    B_img = torch.sign(B_img)
    return B_img


def get_txt_code(txt_model: nn.Module, dataset: CrossModalValidBase, bit, batch_size, drop_integer=False, cuda=True):
    dataset.txt_load()
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=drop_integer)
    B_txt = torch.zeros(len(dataset), bit, dtype=torch.float)
    if cuda:
        B_txt = B_txt.cuda()
    txt_model.eval()
    for data in tqdm(dataloader):
        index = data['index'].numpy()  # type: np.ndarray
        txt = data['txt']  # type: torch.Tensor
        if cuda:
            txt = txt.cuda()

        g = txt_model(txt)
        B_txt[index, :] = g.data
    B_txt = torch.sign(B_txt)
    return B_txt


def precision_recall(query_hash, retrieval_hash, query_label, retrieval_label):
    return calc_precisions_topn(query_hash, retrieval_hash, query_label, retrieval_label)
    # precision = []
    # for i in np.arange(1, 11):
    #     precision.append(calc_precision(query_hash, retrieval_hash, query_label, retrieval_label, recall=0.1 * i))
    # return precision


def pr_value(query_hash, retrieval_hash, query_label, retrieval_label):
    return calc_precisions_hash(query_hash, retrieval_hash, query_label, retrieval_label)


def PR_curve(precisions: np.ndarray, label: list, title: str, x=None):
    # bit = precisions.shape[1]
    from matplotlib import pyplot as plt
    # min_presion = np.min([np.min(l) for l in precisions])
    # max_presion = np.max([np.max(l) for l in precisions])
    min_presion = 0.5
    max_presion = 1
    plt.title(title)
    plt.xticks(np.arange(0.1, 1.1, 0.1))
    plt.xlabel("recall")
    plt.yticks(np.arange(round(min_presion * 10 - 1) * 0.1, (round(max_presion * 10)+1) * 0.1, 0.1))
    plt.ylabel("precision")
    if x is None:
        x = np.arange(0.02, 1.02, 0.02)
        # x = np.expand_dims(x, precisions.shape)
    colors = ['red', 'blue', 'c', 'green', 'yellow', 'black', 'lime', 'grey', 'pink', 'navy']
    markets = ['o', 'v', '^', '>', '<', '+', 'x', '*', 'd', 'D']
    for i in range(precisions.shape[0]):
        # plt.plot(x[i], precisions[i, :], marker=markets[i % 10], color=colors[i % 10], label=label[i])
        plt.plot(x[i], precisions[i], color=colors[i % 10], label=label[i])
        # plt.plot(x, precisions[i, :], color=colors[i % 10], label=label[i])
    plt.grid()
    ax = plt.axes()
    ax.set(xlim=(0, 1), ylim=(round(min_presion * 10 - 1) * 0.1, (round(max_presion * 10)) * 0.1))
    plt.legend()
    # plt.axes('tight')
    plt.show()


def PR_radius_curve(precisions: np.ndarray, label: list, title: str):
    # bit = precisions.shape[1]
    from matplotlib import pyplot as plt
    min_presion = np.min(precisions)
    max_presion = np.max(precisions)
    plt.title(title)
    plt.xticks(np.arange(1, 17, 1))
    x = np.arange(1, 17, 1)
    plt.xlabel("radius")
    plt.yticks(np.arange(round(min_presion * 10) * 0.1 - 0.1, (round(max_presion * 10) + 1) * 0.1 + 0.2, 0.1))
    plt.ylabel("precision")
    colors = ['red', 'blue', 'c', 'green', 'yellow', 'black', 'lime', 'grey', 'pink']
    markets = ['o', 'v', '^', '>', '<', '+', 'x', '*', 'd']
    for i in range(precisions.shape[0]):
        plt.plot(x, precisions[i, :], marker=markets[i % 9], color=colors[i % 9], label=label[i])
    plt.legend()
    plt.show()


def bit_scalable(img_model, txt_model, qB_img, qB_txt, rB_img, rB_txt,  dataset: CrossModalValidBase, to_bit=[64, 32, 16]):

    def get_rank(img_net, txt_net):
        from torch.nn import functional as F
        w_img = img_net.weight
        w_txt = txt_net.weight
        w_img = F.softmax(w_img, dim=0)
        w_txt = F.softmax(w_txt, dim=0)
        w = torch.cat([w_img, w_txt], dim=0)
        w = torch.sum(w, dim=0)
        _, ind = torch.sort(w)
        return ind

    rank_index = get_rank(img_model, txt_model)
    dataset.query()
    query_label = dataset.get_all_label()
    dataset.retrieval()
    retrieval_label = dataset.get_all_label()

    def calc_map(ind):
        qB_img_ind = qB_img[:, ind].cpu()
        qB_txt_ind = qB_txt[:, ind].cpu()
        rB_img_ind = rB_img[:, ind].cpu()
        rB_txt_ind = rB_txt[:, ind].cpu()
        mAPi2t = calc_map_k(qB_img_ind, rB_txt_ind, query_label, retrieval_label)
        mAPt2i = calc_map_k(qB_txt_ind, rB_img_ind, query_label, retrieval_label)
        return mAPi2t, mAPt2i

    print("bit scalable from 128 bit:")
    for bit in to_bit:
        if bit >= 128:
            continue
        bit_ind = rank_index[128 - bit: bit]
        mAPi2t, mAPt2i = calc_map(bit_ind)
        print("%3d: i->t %4.4f| t->i %4.4f" % (bit, mAPi2t, mAPt2i))


