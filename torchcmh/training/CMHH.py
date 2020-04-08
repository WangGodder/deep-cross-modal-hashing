# -*- coding: utf-8 -*-
# @Time    : 2019/7/13
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcmh.models import MLP, cnnf
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor
from torchcmh.dataset import single_data
from torchcmh.loss.distance import euclidean_dist_matrix
from torchcmh.loss.common_loss import focal_loss


class CMHH(TrainBase):
    """
    Cao et al. Cross-modal hamming hashing.
    In The European Conference on Computer Vision (ECCV). September 2018

    Attention: this paper did not give parameters. All parameters may be not right.
    """
    def __init__(self, data_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
        super(CMHH, self).__init__("CMHH", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = single_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ['focal loss', 'quantization loss', 'loss']
        self.parameters = {'gamma': 2, 'alpha': 0.5, 'beta': 0.5, 'lambda': 0}
        self.lr = {'img': 0.02, 'txt': 0.2}
        self.lr_decay_freq = 15
        self.lr_decay = 0.5

        self.num_train = len(self.train_data)
        self.img_model = cnnf.get_cnnf(bit)
        self.txt_model = MLP.MLP(self.train_data.get_tag_length(), bit)
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
        hamming_dist = euclidean_dist_matrix(cur_h, O)
        logit = torch.exp(-hamming_dist * self.parameters['beta'])
        sim = calc_neighbor(label, self.train_label)
        focal_pos = sim * focal_loss(logit, gamma=self.parameters['gamma'], alpha=self.parameters['alpha'])
        focal_neg = (1 - sim) * focal_loss(1 - logit, gamma=self.parameters['gamma'], alpha=1-self.parameters['alpha'])
        fl_loss = torch.mean(focal_pos + focal_neg)
        quantization_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(torch.sign(cur_h) - cur_h, 2), dim=1))) * self.parameters['lambda']
        return fl_loss, quantization_loss


def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
    trainer = CMHH(dataset_name, img_dir, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train()

