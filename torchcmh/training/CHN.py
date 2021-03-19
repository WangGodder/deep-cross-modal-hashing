# -*- coding: utf-8 -*-
# @Time    : 2019/8/4
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
from torch.optim import SGD
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchcmh.models import cnnf, MLP
from torchcmh.training.base import TrainBase
from torchcmh.dataset import single_data


class CHN(TrainBase):
    """
     Jianmin Wang Philip S.Yu Yue Cao,Ming sheng Long. Correlation hashing network for efficient cross-modal retrieval.
     In 28th British Machine Vision Conference.
    """
    def __init__(self, data_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
        super(CHN, self).__init__("CHN", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = single_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ['cosine loss', 'quantization loss', 'loss']
        self.parameters = {'beta': 0.8, 'lambda': 0.01}
        self.lr = {'img': 0.01, 'txt': 1}
        self.lr_decay_freq = 10
        self.lr_decay = 0.5
        self.num_train = len(self.train_data)
        self.img_model = cnnf.get_cnnf(bit)
        self.txt_model = MLP.MLP(self.train_data.get_tag_length(), bit)
        self.F_buffer = torch.randn(self.num_train, bit)
        self.G_buffer = torch.randn(self.num_train, bit)
        self.train_L = self.train_data.get_all_label()
        if cuda:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()
            self.train_L = self.train_L.cuda()
            self.F_buffer = self.F_buffer.cuda()
            self.G_buffer = self.G_buffer.cuda()
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

                cosine_loss, quantization_loss = self.object_fucntion(hash_img, G, label, label)
                loss = cosine_loss + quantization_loss
                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()
                self.remark_loss(cosine_loss, quantization_loss, loss)
            self.print_loss(epoch)
            self.plot_loss("img loss")
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

                cosine_loss, quantization_loss = self.object_fucntion(hash_txt, F, label, label)
                loss = cosine_loss + quantization_loss
                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()
                self.remark_loss(cosine_loss, quantization_loss, loss)
            self.print_loss(epoch)
            self.plot_loss("txt loss")
            self.reset_loss()
            self.valid(epoch)
            self.lr_schedule()
            self.plotter.next_epoch()

    def object_fucntion(self, cur_h, O, label_img, label_txt):
        ones = torch.ones_like(cur_h)
        sim = torch.matmul(label_img, label_txt.t()) > 0
        sim = sim.float()
        sim = sim * 2 - 1
        inner_product = torch.matmul(cur_h, O.t())
        length_img = torch.sqrt(torch.sum(torch.pow(cur_h, 2), dim=1, keepdim=True))
        length_txt = torch.sqrt(torch.sum(torch.pow(O, 2), dim=1, keepdim=True))
        cosine = torch.div(inner_product, torch.matmul(length_img, length_txt.t()))
        cosine_loss = torch.mean(torch.relu_(torch.pow(self.parameters['beta'] - sim * cosine, 2)))
        inner_product_hash = torch.matmul(cur_h, ones.t())
        ones_length_img = torch.ones_like(length_img)
        quantization_img = torch.mean(torch.relu_(self.parameters['beta'] - torch.div(inner_product_hash, torch.matmul(length_img, ones_length_img.t()))))
        quantization_loss = quantization_img * self.parameters['lambda']
        return cosine_loss, quantization_loss


def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
    trainer = CHN(dataset_name, img_dir, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train()