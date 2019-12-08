# -*- coding: utf-8 -*-
# @Time    : 2019/7/11
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcmh.models import cnnf, MLP
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor, AverageMeter
from torchcmh.dataset.utils import single_data


class DCMH(TrainBase):
    """
    Li W J Jiang Q Y. Deep cross-modal hashing
    In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3232– 3240. 2017.
    http://openaccess.thecvf.com/content_cvpr_2017/papers/Jiang_Deep_Cross-Modal_Hashing_CVPR_2017_paper.pdf
    """
    def __init__(self, data_name, img_dir, bit, visdom=True, batch_size=128, cuda=True, **kwargs):
        super(DCMH, self).__init__("DCMH", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = single_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ["log loss", 'quantization loss', 'balance loss', 'loss']
        self.parameters = {'gamma': 1, 'eta': 1}
        self.max_epoch = 500
        self.lr = {'img': 10 ** (-1.5), 'txt': 10 ** (-1.5)}
        self.lr_decay_freq = 1
        self.lr_decay = (1e-6 / 10**(-1.5))**(1/self.max_epoch)
        self.num_train = len(self.train_data)
        self.img_model = cnnf.get_cnnf(bit)
        self.txt_model = MLP.MLP(self.train_data.get_tag_length(), bit, leakRelu=False)
        self.F_buffer = torch.randn(self.num_train, bit)
        self.G_buffer = torch.randn(self.num_train, bit)
        self.train_L = self.train_data.get_all_label()
        self.ones = torch.ones(batch_size, 1)
        self.ones_ = torch.ones(self.num_train - batch_size, 1)
        if cuda:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()
            self.train_L = self.train_L.cuda()
            self.F_buffer = self.F_buffer.cuda()
            self.G_buffer = self.G_buffer.cuda()
            self.ones = self.ones.cuda()
            self.ones_ = self.ones_.cuda()
        self.Sim = calc_neighbor(self.train_L, self.train_L)
        self.B = torch.sign(self.F_buffer + self.G_buffer)
        optimizer_img = SGD(self.img_model.parameters(), lr=self.lr['img'])
        optimizer_txt = SGD(self.txt_model.parameters(), lr=self.lr['txt'])
        self.optimizers = [optimizer_img, optimizer_txt]
        self._init()

    def train(self, num_works=4):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, num_workers=num_works, shuffle=False, pin_memory=True)
        for epoch in range(self.max_epoch):
            self.img_model.train()
            self.txt_model.train()
            self.train_data.img_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                sample_L = data['label']  # type: torch.Tensor
                image = data['img']  # type: torch.Tensor
                if self.cuda:
                    image = image.cuda()
                    sample_L = sample_L.cuda()

                cur_f = self.img_model(image)  # cur_f: (batch_size, bit)
                self.F_buffer[ind, :] = cur_f.data
                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)

                logloss, quantization, balance = self.object_function(cur_f, sample_L, G, F, ind)
                loss = logloss + self.parameters['gamma'] * quantization + self.parameters['eta'] * balance
                loss /= (self.num_train * self.batch_size)

                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()

                self.loss_store['log loss'].update(logloss.item(), (self.batch_size * self.num_train))
                self.loss_store['quantization loss'].update(quantization.item(), (self.batch_size * self.num_train))
                self.loss_store['balance loss'].update(balance.item(), (self.batch_size * self.num_train))
                self.loss_store['loss'].update(loss.item())
            self.print_loss(epoch)
            self.plot_loss("img loss")
            self.reset_loss()

            self.train_data.txt_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                sample_L = data['label']  # type: torch.Tensor
                text = data['txt']  # type: torch.Tensor
                if self.cuda:
                    text = text.cuda()
                    sample_L = sample_L.cuda()

                cur_g = self.txt_model(text)  # cur_g: (batch_size, bit)
                self.G_buffer[ind, :] = cur_g.data
                F = Variable(self.F_buffer)
                G = Variable(self.G_buffer)

                # calculate loss
                logloss, quantization, balance = self.object_function(cur_g, sample_L, F, G, ind)
                loss = logloss + self.parameters['gamma'] * quantization + self.parameters['eta'] * balance
                loss /= (self.num_train * self.batch_size)

                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()

                self.loss_store['log loss'].update(logloss.item(), (self.batch_size * self.num_train))
                self.loss_store['quantization loss'].update(quantization.item(), (self.batch_size * self.num_train))
                self.loss_store['balance loss'].update(balance.item(), (self.batch_size * self.num_train))
                self.loss_store['loss'].update(loss.item())
            self.print_loss(epoch)
            self.plot_loss('text loss')
            self.reset_loss()
            self.B = torch.sign(self.F_buffer + self.G_buffer)
            self.valid(epoch)
            self.lr_schedule()
            self.plotter.next_epoch()
        print("train finish")

    def object_function(self, cur_h: torch.Tensor, sample_label: torch.Tensor, A: torch.Tensor, C: torch.Tensor,ind):
        unupdated_ind = np.setdiff1d(range(self.num_train), ind)
        S = calc_neighbor(sample_label, self.train_L)
        theta = 1.0 / 2 * torch.matmul(cur_h, A.t())
        logloss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta)))
        quantization = torch.sum(torch.pow(self.B[ind, :] - cur_h, 2))
        balance = torch.sum(torch.pow(cur_h.t().mm(self.ones) + C[unupdated_ind].t().mm(self.ones_), 2))
        return logloss, quantization, balance


def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
    trainer = DCMH(dataset_name, img_dir, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train()


def calc_loss(B, F, G, Sim, gamma, eta):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    term1 = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    term2 = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    term3 = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    loss = term1 + gamma * term2 + eta * term3
    return loss
