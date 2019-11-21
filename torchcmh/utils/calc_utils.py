import torch
import numpy as np
from tqdm import tqdm


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


def calc_precisions_hash(qB, rB, query_L, retrieval_L):
    qB = qB.float()
    rB = rB.float()
    qB = torch.sign(qB - 0.5)
    rB = torch.sign(rB - 0.5)
    num_query = query_L.shape[0]
    num_retrieval = retrieval_L.shape[0]
    bit = qB.shape[1]
    hamm = calc_hammingDist(qB, rB)
    hamm = hamm.type(torch.ByteTensor)
    total_num = [0] * (bit + 1)
    max_hamm = int(torch.max(hamm))
    gnd = (query_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze()
    total_right = torch.sum(torch.matmul(query_L, retrieval_L.t())>0)
    precisions = np.zeros([max_hamm + 1])
    recalls = np.zeros([max_hamm + 1])
    # _, index = torch.sort(hamm)
    # del _
    # for i in range(index.shape[0]):
    #     gnd[i, :] = gnd[i, index[i]]
    # del index
    right_num = 0
    recall_num = 0
    for i, radius in enumerate(range(0, max_hamm+1)):
        recall = torch.nonzero(hamm == radius)
        right = gnd[recall.split(1, dim=1)]
        recall_num += recall.shape[0]
        del recall
        right_num += torch.nonzero(right).shape[0]
        del right
        precisions[i] += (right_num / (recall_num + 1e-8))
        # recalls[i] += (recall_num / num_retrieval / num_query)
        recalls[i] += (recall_num / total_right)
    return precisions, recalls


def calc_precisions_hamming_radius(qB, rB, query_L, retrieval_L, hamming_gas=1):
    num_query = query_L.shape[0]
    bit = qB.shape[1]
    precisions = [0] * int(bit / hamming_gas)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)  # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        for i, recall in enumerate(np.arange(1, bit+1, hamming_gas)):
            total = torch.nonzero(hamm <= recall).squeeze().shape[0]
            if total == 0:
                precisions[i] += 0
                continue
            right = torch.nonzero(gnd[: total]).squeeze().numpy()
            right_num = right.size

            precisions[i] += (right_num / total)
    for i in range(len(precisions)):
        precisions[i] /= num_query
    return precisions


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = label1.matmul(label2.transpose(0, 1)) > 0
    return Sim.float()


def norm_max_min(x: torch.Tensor, dim=None):
    if dim is None:
        max = torch.max(x)
        min = torch.min(x)
    if dim is not None:
        max = torch.max(x, dim=dim)[0]
        min = torch.min(x, dim=dim)[0]
        if dim > 0:
            max = max.unsqueeze(len(x.shape) - 1)
            min = min.unsqueeze(len(x.shape) - 1)
    norm = (x - min) / (max - min)
    return norm


def norm_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = (x - mean) / std
    return norm


def norm_abs_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = torch.abs(x - mean) / std
    return norm


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def calc_IF(all_bow):
    word_num = torch.sum(all_bow, dim=0)
    total_num = torch.sum(word_num)
    IF = word_num / total_num
    return IF


# def calc_loss(B, F, G, Sim, gamma1, gamma2, eta):
#     theta = torch.matmul(F, G.transpose(0, 1)) / 2
#     inter_loss = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
#     theta_f = torch.matmul(F, F.transpose(0, 1)) / 2
#     intra_img = torch.sum(torch.log(1 + torch.exp(theta_f)) - Sim * theta_f)
#     theta_g = torch.matmul(G, G.transpose(0, 1)) / 2
#     intra_txt = torch.sum(torch.log(1 + torch.exp(theta_g)) - Sim * theta_g)
#     intra_loss = gamma1 * intra_img + gamma2 * intra_txt
#     quan_loss = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2)) * eta
#     # term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
#     # loss = term1 + gamma * term2 + eta * term3
#     loss = inter_loss + intra_loss + quan_loss
#     return loss


# if __name__ == '__main__':
#     qB = torch.Tensor([[1, -1, 1, 1],
#                        [-1, -1, -1, 1],
#                        [1, 1, -1, 1],
#                        [1, 1, 1, -1]])
#     rB = torch.Tensor([[1, -1, 1, -1],
#                        [-1, -1, 1, -1],
#                        [-1, -1, 1, -1],
#                        [1, 1, -1, -1],
#                        [-1, 1, -1, -1],
#                        [1, 1, -1, 1]])
#     query_L = torch.Tensor([[0, 1, 0, 0],
#                             [1, 1, 0, 0],
#                             [1, 0, 0, 1],
#                             [0, 1, 0, 1]])
#     retrieval_L = torch.Tensor([[1, 0, 0, 1],
#                                 [1, 1, 0, 0],
#                                 [0, 1, 1, 0],
#                                 [0, 0, 1, 0],
#                                 [1, 0, 0, 0],
#                                 [0, 0, 1, 0]])
#
#     map = calc_map_k(qB, rB, query_L, retrieval_L)
#     print(map)
