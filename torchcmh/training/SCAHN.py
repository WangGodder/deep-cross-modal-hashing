# -*- coding: utf-8 -*-
# @Time    : 2019/7/13
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcmh.models.SCAHN import resnet18, resnet34, get_MS_Text
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor
from torchcmh.utils import calc_map_k
from torchcmh.dataset import pairwise_data

__all__ = ['train']

"""
@article{SCAHN
author = {Xinzhi Wang,Xitao Zou,Erwin M. Bakker, Song Wu},
title = {Self-Constraining and Attention-based Hashing Network for Bit-Scalable Cross-Modal Retrieval},
journaltitle = {Neurocomputing},
date = {13 March 2020},
}
"""


class SCAHN(TrainBase):
    def __init__(self, data_name: str, img_dir: str, bit: int, img_net, visdom=True, batch_size=128, cuda=True,
                 **kwargs):
        super(SCAHN, self).__init__("SCAHN", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = pairwise_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ['inter loss', 'intra loss', 'pairwise intra loss', 'quantization loss', 'loss']
        self.parameters = {'fusion num': 4, 'alpha': 2 ** np.log2(bit / 32), 'lambda': 1, 'gamma': 0.01, 'beta': 2}
        self.lr = {'img': 10 ** (-1.1), 'txt': 10 ** (-1.1)}
        self.max_epoch = 500
        self.lr_decay_freq = 1
        self.lr_decay = (10 ** (-6) / 10 ** (-1.5)) ** (1 / self.max_epoch)

        self.num_train = len(self.train_data)
        self.img_model = img_net(bit, self.parameters['fusion num'])
        self.txt_model = get_MS_Text(self.train_data.get_tag_length(), bit, self.parameters['fusion num'])
        self.train_L = self.train_data.get_all_label()
        self.F_buffer = torch.randn(self.num_train, bit)
        self.G_buffer = torch.randn(self.num_train, bit)
        if cuda:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()
            self.train_L = self.train_L.cuda()
            self.F_buffer = self.F_buffer.cuda()
            self.G_buffer = self.G_buffer.cuda()
        self.B = torch.sign(self.F_buffer + self.G_buffer)

        optimizer_img = SGD(self.img_model.parameters(), lr=self.lr['img'])
        optimizer_txt = SGD(self.txt_model.parameters(), lr=self.lr['txt'])
        self.optimizers = [optimizer_img, optimizer_txt]
        self._init()

    def train(self, num_works=4):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, num_workers=num_works,
                                  shuffle=False, pin_memory=True)
        for epoch in range(self.max_epoch):
            self.img_model.train()
            self.txt_model.train()
            self.train_data.img_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind1 = data['index'].numpy()
                ind2 = data['ano_index'].numpy()

                sample_L1 = data['label']  # type: torch.Tensor
                sample_L2 = data['ano_label']  # type: torch.Tensor
                image1 = data['img']  # type: torch.Tensor
                image2 = data['ano_img']  # type: torch.Tensor
                if self.cuda:
                    image1 = image1.cuda()
                    image2 = image2.cuda()
                    sample_L1 = sample_L1.cuda()
                    sample_L2 = sample_L2.cuda()

                # get output from img net
                middle_hash1, hash1 = self.img_model(image1)
                middle_hash2, hash2 = self.img_model(image2)

                hash1_layers = middle_hash1
                hash2_layers = middle_hash2
                hash1 = torch.tanh(hash1)
                hash2 = torch.tanh(hash2)
                hash1_layers.append(hash1)
                hash2_layers.append(hash2)

                self.F_buffer[ind1, :] = hash1.data
                self.F_buffer[ind2, :] = hash2.data
                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)

                inter_loss, intra_loss, intra_pair_loss, quantization_loss = self.object_function(hash1_layers,
                                                                                                  hash2_layers, hash1,
                                                                                                  hash2, sample_L1,
                                                                                                  sample_L2, F, G, ind1,
                                                                                                  ind2)

                loss = inter_loss + intra_pair_loss + intra_loss + quantization_loss

                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()
                self.remark_loss(inter_loss, intra_loss, intra_pair_loss, quantization_loss, loss)
            self.print_loss(epoch)
            self.plot_loss("img loss")
            self.reset_loss()
            weight = self.img_model.weight.weight  # type: torch.Tensor
            weight = torch.mean(weight, dim=1)
            for i in range(weight.shape[0]):
                self.plotter.plot("img ms weight", 'part' + str(i), weight[i].item())

            self.train_data.txt_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind1 = data['index'].numpy()
                ind2 = data['ano_index'].numpy()
                sample_L1 = data['label']  # type: torch.Tensor
                sample_L2 = data['ano_label']  # type: torch.Tensor
                tag1 = data['txt']  # type: torch.Tensor
                tag2 = data['ano_txt']  # type: torch.Tensor
                if self.cuda:
                    tag1 = tag1.cuda()
                    tag2 = tag2.cuda()
                    sample_L1 = sample_L1.cuda()
                    sample_L2 = sample_L2.cuda()

                # get output from img net
                middle_hash1, hash1 = self.txt_model(tag1)
                middle_hash2, hash2 = self.txt_model(tag2)

                hash1_layers = middle_hash1
                hash2_layers = middle_hash2
                hash1 = torch.tanh(hash1)
                hash2 = torch.tanh(hash2)
                hash1_layers.append(hash1)
                hash2_layers.append(hash2)

                self.G_buffer[ind1, :] = hash1.data
                self.G_buffer[ind2, :] = hash2.data
                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)

                inter_loss, intra_loss, intra_pair_loss, quantization_loss = self.object_function(hash1_layers,
                                                                                                  hash2_layers, hash1,
                                                                                                  hash2,
                                                                                                  sample_L1, sample_L2,
                                                                                                  G, F, ind1, ind2)

                loss = inter_loss + intra_pair_loss + intra_loss + quantization_loss

                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()
                self.remark_loss(inter_loss, intra_loss, intra_pair_loss, quantization_loss, loss)
            self.print_loss(epoch)
            self.plot_loss("txt loss")
            self.reset_loss()
            weight = self.txt_model.weight.weight  # type: torch.Tensor
            weight = torch.mean(weight, dim=1)
            for i in range(weight.shape[0]):
                self.plotter.plot("txt ms weight", 'part' + str(i), weight[i].item())
            self.B = torch.sign(self.F_buffer + self.G_buffer)
            self.valid(epoch)
            self.lr_schedule()
            self.plotter.next_epoch()

    def object_function(self, hash_layers1, hash_layer2, final_hash1, final_hash2, label1, label2, F, G, ind1, ind2):
        # get similarity matrix
        S_inter1 = calc_neighbor(label1, self.train_L)
        S_inter2 = calc_neighbor(label2, self.train_L)
        S_intra = calc_neighbor(label1, label2)

        inter_loss1, inter_loss2 = calc_inter_loss(hash_layers1, hash_layer2, S_inter1, S_inter2, G,
                                                   self.parameters['alpha'])
        inter_loss = 0.5 * (inter_loss1 + inter_loss2)
        intra_loss1, intra_loss2 = calc_intra_loss(hash_layers1, hash_layer2, S_inter1, S_inter2, F,
                                                   self.parameters['alpha'])
        intra_loss = 0.5 * (intra_loss1 + intra_loss2) * self.parameters['lambda']
        intra_pair_loss = calc_intra_pairwise_loss(hash_layers1, hash_layer2, S_intra, self.parameters['alpha'])
        intra_pair_loss = intra_pair_loss * self.parameters['gamma']
        quantization_loss1 = torch.mean(torch.sum(torch.pow(self.B[ind1, :] - final_hash1, 2), dim=1))
        quantization_loss2 = torch.mean(torch.sum(torch.pow(self.B[ind2, :] - final_hash2, 2), dim=1))
        quantization_loss = 0.5 * (quantization_loss1 + quantization_loss2) / self.bit * self.parameters['beta']
        return inter_loss, intra_loss, intra_pair_loss, quantization_loss

    @staticmethod
    def bit_scalable(img_model, txt_model, qB_img, qB_txt, rB_img, rB_txt, dataset, to_bit=[64, 32, 16]):
        def get_rank(img_net, txt_net):
            from torch.nn import functional as F
            w_img = img_net.weight.weight
            w_txt = txt_net.weight.weight
            # w_img = F.softmax(w_img, dim=0)
            # w_txt = F.softmax(w_txt, dim=0)
            w = torch.cat([w_img, w_txt], dim=0)
            w = torch.sum(w, dim=0)
            # _, ind = torch.sort(w)
            _, ind = torch.sort(w, descending=True)  # 临时降序
            return ind

        hash_length = qB_img.size(1)
        rank_index = get_rank(img_model, txt_model)
        dataset.query()
        query_label = dataset.get_all_label()
        dataset.retrieval()
        retrieval_label = dataset.get_all_label()

        def calc_map(ind):
            qB_img_ind = qB_img[:, ind]
            qB_txt_ind = qB_txt[:, ind]
            rB_img_ind = rB_img[:, ind]
            rB_txt_ind = rB_txt[:, ind]
            mAPi2t = calc_map_k(qB_img_ind, rB_txt_ind, query_label, retrieval_label)
            mAPt2i = calc_map_k(qB_txt_ind, rB_img_ind, query_label, retrieval_label)
            return mAPi2t, mAPt2i

        print("bit scalable from 128 bit:")
        for bit in to_bit:
            if bit >= hash_length:
                continue
            bit_ind = rank_index[hash_length - bit: hash_length]
            mAPi2t, mAPt2i = calc_map(bit_ind)
            print("%3d: i->t %4.4f| t->i %4.4f" % (bit, mAPi2t, mAPt2i))

    def valid(self, epoch):
        """
        valid current training model, and save the best model and hash code.
        :param epoch: current epoch
        :return:
        """
        mapi2t, mapt2i, qB_img, qB_txt, rB_img, rB_txt = \
            self.valid_calc(self.img_model, self.txt_model, self.valid_data, self.bit, self.batch_size,
                            return_hash=True)
        if mapt2i + mapi2t >= self.max_mapi2t + self.max_mapt2i:
            self.max_mapi2t = mapi2t
            self.max_mapt2i = mapt2i
            self.best_epoch = epoch
            import os
            self.img_model.save_dict(
                os.path.join(self.checkpoint_dir, str(self.bit) + '-' + self.img_model.module_name + '.pth'))
            self.txt_model.save_dict(
                os.path.join(self.checkpoint_dir, str(self.bit) + '-' + self.txt_model.module_name + '.pth'))
            self.qB_img = qB_img.cpu()
            self.qB_txt = qB_txt.cpu()
            self.rB_img = rB_img.cpu()
            self.rB_txt = rB_txt.cpu()
            if (epoch + 1) > 10:
                self.bit_scalable(self.img_model, self.txt_model, self.qB_img, self.qB_txt, self.rB_img, self.rB_txt,
                                  self.valid_data)
            # self.best_train_img, self.best_train_txt = self.get_train_hash()
        print(
            'epoch: [%3d/%3d], valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f in epoch %d' %
            (epoch + 1, self.max_epoch, mapi2t, mapt2i, self.max_mapi2t, self.max_mapt2i, self.best_epoch + 1))
        if self.plotter:
            self.plotter.plot("mAP", 'i->t', mapi2t.item())
            self.plotter.plot("mAP", "t->i", mapt2i.item())
        self.save_code(epoch)


def train(dataset_name: str, img_dir: str, bit: int, img_net_name='resnet34', visdom=True, batch_size=128, cuda=True,
          **kwargs):
    img_net = resnet34 if img_net_name == 'resnet34' else resnet18
    trainer = SCAHN(dataset_name, img_dir, bit, img_net, visdom, batch_size, cuda, **kwargs)
    trainer.train()


def calc_inter_loss(hash1_layers, hash2_layers, S1, S2, O, alpha):
    inter_loss1 = 0
    for index, hash1_layer in enumerate(hash1_layers):
        theta = 1.0 / alpha * torch.matmul(hash1_layer, O.t())
        logloss = -torch.mean(S1 * theta - torch.log(1 + torch.exp(theta)))
        if torch.isinf(logloss):
            print("the log loss is inf in hash1 of layer %d, with the max of theta is %3.4f" % (
                index, torch.max(theta).data))
        inter_loss1 += logloss
    inter_loss2 = 0
    for index, hash2_layer in enumerate(hash2_layers):
        theta = 1.0 / alpha * torch.matmul(hash2_layer, O.t())
        logloss = -torch.mean(S2 * theta - torch.log(1 + torch.exp(theta)))
        if torch.isinf(logloss):
            print("the log loss is inf in hash2 of layer %d, with the max of theta is %3.4f" % (
                index, torch.max(theta).data))
        inter_loss2 += logloss
    return inter_loss1 / len(hash1_layers), inter_loss2 / len(hash2_layers)


def calc_intra_pairwise_loss(hash1_layers, hash2_layers, S, alpha):
    intra_loss = 0
    for index in range(len(hash1_layers)):
        hash1_layer = hash1_layers[index]
        hash2_layer = hash2_layers[index]
        theta = 1.0 / alpha * torch.matmul(hash1_layer, hash2_layer.t())
        logloss = -torch.mean(S * theta - torch.log(1 + torch.exp(theta)))
        if torch.isinf(logloss):
            print("the log loss is inf in hash1 and hash2 of layer %d, with the max of theta is %3.4f" % (
                index, torch.max(theta).data))
        intra_loss += logloss
    return intra_loss / len(hash1_layers)


def calc_intra_loss(hash1_layers, hash2_layers, S1, S2, O, alpha):
    intra_loss1 = 0
    for hash in hash1_layers:
        theta = 1.0 / alpha * torch.matmul(hash, O.t())
        logloss = -torch.mean(S1 * theta - torch.log(1 + torch.exp(theta)))
        intra_loss1 += logloss
    intra_loss2 = 0
    for hash in hash2_layers:
        theta = 1.0 / alpha * torch.matmul(hash, O.t())
        logloss = -torch.mean(S2 * theta - torch.log(1 + torch.exp(theta)))
        intra_loss2 += logloss
    return intra_loss1 / len(hash1_layers), intra_loss2 / len(hash2_layers)


def calc_loss(B, F, G, Sim, gamma1, gamma2, eta, alpha):
    theta = torch.matmul(F, G.transpose(0, 1)) / alpha
    inter_loss = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    theta_f = torch.matmul(F, F.transpose(0, 1)) / alpha
    intra_img = torch.sum(torch.log(1 + torch.exp(theta_f)) - Sim * theta_f)
    theta_g = torch.matmul(G, G.transpose(0, 1)) / alpha
    intra_txt = torch.sum(torch.log(1 + torch.exp(theta_g)) - Sim * theta_g)
    intra_loss = gamma1 * intra_img + gamma2 * intra_txt
    quan_loss = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2)) * eta
    # term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    # loss = term1 + gamma * term2 + eta * term3
    loss = inter_loss + intra_loss + quan_loss
    return loss
