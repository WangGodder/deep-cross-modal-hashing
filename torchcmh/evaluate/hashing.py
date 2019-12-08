# -*- coding: utf-8 -*-
# @Time    : 2019/12/2
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
__all__ = ['calc_map_k', 'calc_precisions_topn']


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    qB = torch.sign(qB)
    rB = torch.sign(rB)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_precisions_topn(qB, rB, query_L, retrieval_L, recall_gas=0.02, num_retrieval=10000):
    qB = qB.float()
    rB = rB.float()
    qB = torch.sign(qB - 0.5)
    rB = torch.sign(rB - 0.5)
    num_query = query_L.shape[0]
    # num_retrieval = retrieval_L.shape[0]
    precisions = [0] * int(1 / recall_gas)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        for i, recall in enumerate(np.arange(recall_gas, 1 + recall_gas, recall_gas)):
            total = int(num_retrieval * recall)
            right = torch.nonzero(gnd[: total]).squeeze().numpy()
            # right_num = torch.nonzero(gnd[: total]).squeeze().shape[0]
            right_num = right.size
            precisions[i] += (right_num/total)
    for i in range(len(precisions)):
        precisions[i] /= num_query
    return precisions
