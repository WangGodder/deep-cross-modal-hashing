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
from torchcmh.models import alexnet, mlp, vgg_f
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor, AverageMeter
from torchcmh.utils import get_plotter
from torchcmh.training.valid import valid
from torchcmh.dataset.utils import single_data
from torchcmh.loss.distance import focal_loss, euclidean_dist
import os


class CMHH(TrainBase):
    def __init__(self, data_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
        super(CMHH, self).__init__("CMHH", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = single_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ['focal loss', 'quantization loss', 'loss']
        self.parameters = {'gamma': 2, 'alpha': 0.5, 'beta': 0.5, 'lambda': 0}
        self.lr = {'img': 0.02, 'txt': 0.2}
        self.lr_decay_freq = 15
        self.lr_decay = 0.5

        self.num_train = len(self.train_data)
        self.img_model = vgg_f.get_vgg_f(bit)
        self.txt_model = mlp.MLP(self.train_data.get_tag_length(), bit)
        self.train_label = self.train_data.get_all_label()
        self.F_buffer = torch.randn(self.num_train, bit)
        self.G_buffer = torch.randn(self.num_train, bit)
        if cuda:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()
            self.train_label = self.train_label.cuda()
            self.F_buffer = self.F_buffer.cuda()
            self.G_buffer = self.G_buffer.cuda()
        self.B = torch.sign(self.F_buffer + self.G_buffer)

        optimizer_img = SGD(self.img_model.parameters(), lr=self.lr['img'], momentum=0.9)
        optimizer_txt = SGD(self.txt_model.parameters(), lr=self.lr['txt'], momentum=0.9)
        self.optimizers = [optimizer_img, optimizer_txt]
        self._init()

    def train(self, num_works=4):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, num_workers=num_works, shuffle=False,
                                  pin_memory=True)
        for epoch in range(self.max_epoch):
            self.img_model.train()
            self.txt_model.train()
            self.train_data.img_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                img = data['img']
                label = data['label']
                if self.cuda:
                    img = img.cuda()
                    label = label.cuda()
                hash_img = self.img_model(img)
                hash_img = torch.tanh(hash_img)
                self.F_buffer[ind] = hash_img.data
                G = Variable(self.G_buffer)

                fl_loss, quantization_loss = self.object_function(hash_img, G, label, ind)
                loss = fl_loss + quantization_loss
                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()
                self.remark_loss(fl_loss, quantization_loss, loss)
            self.print_loss(epoch)
            self.plot_loss('img loss')
            self.reset_loss()
            self.train_data.txt_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                txt = data['txt']
                label = data['label']
                if self.cuda:
                    txt = txt.cuda()
                    label = label.cuda()
                hash_txt = self.txt_model(txt)
                hash_txt = torch.tanh(hash_txt)
                self.G_buffer[ind] = hash_txt.data
                F = Variable(self.F_buffer)

                fl_loss, quantization_loss = self.object_function(hash_txt, F, label, ind)
                loss = fl_loss + quantization_loss
                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()
                self.remark_loss(fl_loss, quantization_loss, loss)
            self.B = torch.sign(self.F_buffer + self.G_buffer)
            self.print_loss(epoch)
            self.plot_loss('txt loss')
            self.reset_loss()
            self.valid(epoch)
            self.lr_schedule()
            self.plotter.next_epoch()

    def object_function(self, cur_h, O, label, ind):
        hamming_dist = euclidean_dist(cur_h, O)
        logit = torch.exp(-hamming_dist * self.parameters['beta'])
        sim = calc_neighbor(label, self.train_label)
        focal_pos = sim * focal_loss(logit, gamma=self.parameters['gamma'], alpha=self.parameters['alpha'])
        focal_neg = (1 - sim) * focal_loss(1 - logit, gamma=self.parameters['gamma'], alpha=1-self.parameters['alpha'])
        fl_loss = torch.mean(focal_pos + focal_neg)
        # fl_loss = torch.mean(torch.log(1 + torch.exp(hamming_dist)) - sim * hamming_dist)
        # quantization_img = torch.mean(torch.sum(torch.pow(torch.abs(hash_img) - 1, 2), dim=1))
        # quantization_txt = torch.mean(torch.sum(torch.pow(torch.abs(hash_txt) - 1, 2), dim=1))
        quantization_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(torch.sign(cur_h) - cur_h, 2), dim=1))) * self.parameters['lambda']
        return fl_loss, quantization_loss


def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
    trainer = CMHH(dataset_name, img_dir, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train()


# def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True):
#     lr = 10 ** (-2.0)
#     end_lr = 10 ** (-5.0)
#     max_epoch = 500
#     gamma = 2
#     beta = 0.25
#     lambdaa = 1
#     print("training %s, hyper-paramter list:\n gamma = %3.2f \n beta = %3.2f \n lambda = %3.2f" % (name, gamma, beta, lambdaa))
#     checkpoint_dir = os.path.join('..', 'checkpoints', name, dataset_name)
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#
#     if visdom:
#         plotter = get_plotter(name)
#
#     train_data, valid_data = single_data(dataset_name, img_dir, batch_size=batch_size)
#     num_train = len(train_data)
#
#     img_model = vgg_f.get_vgg_f(bit)
#     txt_model = mlp.MLP(train_data.get_tag_length(), bit)
#
#     F = torch.randn(num_train, bit)
#     G = torch.randn(num_train, bit)
#     all_label = train_data.get_all_label()
#     if cuda:
#         img_model = img_model.cuda()
#         txt_model = txt_model.cuda()
#         F = F.cuda()
#         G = G.cuda()
#         all_label = all_label.cuda()
#     S = calc_neighbor(all_label, all_label)
#     B = torch.sign(F + G)
#
#     optimizer_img = SGD(img_model.parameters(), lr=lr, momentum=0.9)
#     optimizer_txt = SGD(txt_model.parameters(), lr=lr, momentum=0.9)
#
#     learning_rate = np.linspace(lr, end_lr, max_epoch + 1)
#
#     max_mapi2t = max_mapt2i = 0.
#     train_data.both_load()
#     train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=4, shuffle=False, pin_memory=True)
#     for epoch in range(max_epoch):
#         img_model.train()
#         txt_model.train()
#         train_data.re_random_item()
#         for data in tqdm(train_loader):
#             ind = data['index'].numpy()
#             img = data['img']
#             txt = data['txt']
#             label = data['label']
#
#             if cuda:
#                 img = img.cuda()
#                 txt = txt.cuda()
#                 label = label.cuda()
#
#             sim = calc_neighbor(label, label)
#
#             h_x = img_model(img)
#             h_y = txt_model(txt)
#             h_x = torch.tanh(h_x)
#             h_y = torch.tanh(h_y)
#
#             F[ind] = h_x.data
#             G[ind] = h_y.data
#
#             hamming_dist = 0.5 * (bit - torch.mm(h_x, h_y.t()))
#             hamming_dist *= beta
#             # focal_pos = sim * torch.pow((1 - torch.exp(-hamming_dist)), gamma) * hamming_dist
#             # focal_neg = (1 - sim) * torch.pow(torch.exp(-hamming_dist), gamma) * torch.log(1 - torch.exp(-hamming_dist))
#             focal_pos = sim * focal_loss(torch.exp(-hamming_dist), gamma)
#             focal_neg = (1 - sim) * focal_loss(1 - torch.exp(-hamming_dist), gamma)
#             f_loss = torch.sum(focal_pos + focal_neg) / (batch_size * batch_size)
#
#             collective = torch.sum(torch.pow(torch.abs(h_x) - 1, 2)) + torch.sum(torch.pow(torch.abs(h_y) - 1, 2))
#             collective /= batch_size
#
#             loss = f_loss + lambdaa * collective
#
#             optimizer_img.zero_grad()
#             optimizer_txt.zero_grad()
#             loss.backward()
#             optimizer_img.step()
#             optimizer_txt.step()
#
#             focal_loss_store.update(f_loss.item())
#             collective_loss_store.update(collective.item())
#             loss_store.update(loss.item())
#         print("focal loss: %4.3f, collective loss: %4.3f, loss: %4.3f" % (focal_loss_store.avg, collective_loss_store.avg, loss_store.avg))
#         if visdom:
#             plotter.plot("loss", "focal loss", focal_loss_store.avg)
#             plotter.plot("loss", "collective", collective_loss_store.avg)
#             plotter.plot("loss", "loss", loss_store.avg)
#
#         focal_loss_store.reset()
#         focal_pos_store.reset()
#         focal_neg_store.reset()
#         collective_loss_store.reset()
#         loss_store.reset()
#
#         # train_data.re_random_item()
#         # train_data.txt_load()
#         # for data in tqdm(train_loader):
#         #     ind = data['index'].numpy()
#         #     txt = data['txt']
#         #     label = data['label']
#         #
#         #     if cuda:
#         #         txt = txt.cuda()
#         #         label = label.cuda()
#         #
#         #     sim = calc_neighbor(label, all_label)
#         #
#         #     h_y = txt_model(txt)
#         #     h_y = torch.tanh(h_y)
#         #
#         #     G_buffer[ind] = h_y.data
#         #     F = Variable(F_buffer)
#         #     G = Variable(G_buffer)
#         #
#         #     hamming_dist = 0.5 * (bit - torch.mm(h_y, F.t()))
#         #     hamming_dist *= beta
#         #     focal_pos = sim * torch.pow((1 - torch.exp(-hamming_dist)), gamma) * hamming_dist
#         #     focal_neg = (1 - sim) * torch.pow(torch.exp(-hamming_dist), gamma) * torch.log(1 - torch.exp(-hamming_dist))
#         #     f_loss = torch.sum(focal_pos - focal_neg) / (batch_size * num_train)
#         #     collective = torch.sqrt(torch.sum(torch.pow(torch.abs(h_y) - 1, 2))) / batch_size
#         #
#         #     loss = f_loss + lambdaa * collective
#         #
#         #     optimizer_txt.zero_grad()
#         #     loss.backward()
#         #     optimizer_txt.step()
#         #
#         #     focal_loss_store.update(f_loss.item())
#         #     collective_loss_store.update(collective.item())
#         #     loss_store.update(loss.item())
#         # print("focal loss: %4.3f, collective loss: %4.3f, loss: %4.3f" % (focal_loss_store.avg, collective_loss_store.avg, loss_store.avg))
#         # if plotter is not None:
#         #     plotter.plot("txt loss", "focal loss", focal_loss_store.avg)
#         #     plotter.plot("txt loss", "collective", collective_loss_store.avg)
#         #     plotter.plot("txt loss", "loss", loss_store.avg)
#         #
#         # focal_loss_store.reset()
#         # focal_pos_store.reset()
#         # focal_neg_store.reset()
#         # collective_loss_store.reset()
#         # loss_store.reset()
#
#         loss = calc_loss(F, G, S, gamma, lambdaa, beta)
#         if visdom:
#             plotter.plot("total loss", "loss", loss.item())
#         print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.data, lr))
#
#         mapi2t, mapt2i = valid(img_model, txt_model, valid_data, bit, batch_size)
#         if mapt2i + mapi2t >= max_mapi2t + max_mapt2i:
#             max_mapi2t = mapi2t
#             max_mapt2i = mapt2i
#             torch.save(img_model, os.path.join(checkpoint_dir, str(bit) + 'img' + '.pth'))
#             torch.save(txt_model, os.path.join(checkpoint_dir, str(bit) + 'txt' + '.pth'))
#         print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' %
#               (epoch + 1, mapi2t, mapt2i, max_mapi2t, max_mapt2i))
#         if visdom:
#             plotter.plot("mAP", 'i->t', mapi2t.item())
#             plotter.plot("mAP", "t->i", mapt2i.item())
#
#         lr = learning_rate[epoch + 1]
#         for param in optimizer_img.param_groups:
#             param['lr'] = lr
#         for param in optimizer_txt.param_groups:
#             param['lr'] = lr
#
#         plotter.next_epoch()
#
#
# def calc_loss(F, G, S, gamma, lambdaa, beta):
#     hamming_dist = calc_hammingDist(F, G)
#     hamming_dist *= beta
#     focal_pos = S * torch.pow(1 - torch.exp(-hamming_dist), gamma) * hamming_dist
#     focal_neg = (1 - S) * torch.pow(torch.exp(-hamming_dist), gamma) * torch.log(1 - torch.exp(-hamming_dist))
#     focal_loss = torch.sum(focal_pos + focal_neg)
#     collective = torch.sqrt(torch.sum(torch.pow(torch.abs(F) - 1, 2))) + torch.sqrt(torch.sum(torch.pow(torch.abs(G) - 1, 2)))
#     loss = focal_loss + lambdaa * collective
#     return loss
