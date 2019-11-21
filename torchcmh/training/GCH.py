# -*- coding: utf-8 -*-
# @Time    : 2019/7/11
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
from torch.autograd import Variable
from torch.optim import SGD
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcmh.models.GCH import image_net, enbedding_net, GC_net
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor
from torchcmh.dataset import single_data


class GCH(TrainBase):
    def __init__(self, data_name, img_dir, bit, visdom=True, batch_size=128, cuda=True, **kwargs):
        super(GCH, self).__init__("GCH", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = single_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ["log cross-entropy loss", 'label loss', 'hash loss', 'total loss']
        self.parameters = {'alpha': 1, 'beta': 1, 'gamma': 0.8}
        self.max_epoch = 500
        self.lr = {'img': 10 ** (-1.5), 'txt': 10 ** (-1.5), 'lab': 10 ** (-1.5), 'gcn': 10 ** (-1.5)}
        self.lr_decay_freq = 1
        self.lr_decay = (1e-6 / 10**(-1.5))**(1/self.max_epoch)
        self.num_train = len(self.train_data)
        self.train_L = self.train_data.get_all_label()  # type: torch.Tensor
        self.img_model = image_net.get_image_net(bit, self.train_L.size(1))
        self.txt_model = enbedding_net.get_txt_net(self.train_data.get_tag_length(), bit, self.train_L.size(1))
        self.lab_model = enbedding_net.get_label_net(self.train_L.size(1), bit)
        self.gcn_model = GC_net.GCN(bit, self.train_L.size(1))
        self.hash_buffer = torch.randn(self.num_train, bit)
        self.img_feature_buffer = torch.zeros(self.num_train, bit)
        self.txt_feature_buffer = torch.zeros(self.num_train, bit)
        if cuda:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()
            self.lab_model = self.lab_model.cuda()
            self.gcn_model = self.gcn_model.cuda()
            self.train_L = self.train_L.cuda()
            self.hash_buffer = self.hash_buffer.cuda()
            self.img_feature_buffer = self.img_feature_buffer.cuda()
            self.txt_feature_buffer = self.txt_feature_buffer.cuda()
        self.Sim = calc_neighbor(self.train_L, self.train_L)
        optimizer_img = SGD(self.img_model.parameters(), lr=self.lr['img'])
        optimizer_txt = SGD(self.txt_model.parameters(), lr=self.lr['txt'])
        optimizer_lab = SGD(self.lab_model.parameters(), lr=self.lr['lab'])
        optimizer_gcn = SGD(self.gcn_model.parameters(), lr=self.lr['gcn'])
        self.optimizers = [optimizer_img, optimizer_txt, optimizer_lab, optimizer_gcn]
        self._init()

    def train(self, num_works=4):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, num_workers=num_works, shuffle=False, pin_memory=True)
        for epoch in range(self.max_epoch):
            self.img_model.train()
            self.txt_model.train()
            # label net
            self.train_data.txt_load()
            self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                label = data['label']  # type: torch.Tensor]
                if self.cuda:
                    label = label.cuda()
                hash_representation, label_predict = self.lab_model(label)
                self.hash_buffer[ind, :] = hash_representation.data
                H = Variable(self.hash_buffer)
                logloss, label_loss = self.label_loss(hash_representation, H, label_predict, ind)
                loss = logloss + label_loss
                self.optimizers[2].zero_grad()
                loss.backward()
                self.optimizers[2].step()
                self.remark_loss(logloss.item(), label_loss.item(), 0, loss.item())
            self.print_loss(epoch)
            self.plot_loss("label")
            self.reset_loss()

            # img net
            self.train_data.img_load()
            # self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                image = data['img']  # type: torch.Tensor
                if self.cuda:
                    image = image.cuda()
                hash_representation, label_predict = self.img_model(image)
                self.img_feature_buffer[ind] = hash_representation.data
                H = Variable(self.hash_buffer)
                logloss, hash_loss, label_loss = self.modality_loss(hash_representation, H, label_predict, ind)
                loss = logloss + hash_loss + label_loss
                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()
                self.remark_loss(logloss.item(), hash_loss.item(), label_loss.item(), loss.item())
            self.print_loss(epoch)
            self.plot_loss('image')
            self.reset_loss()

            self.train_data.txt_load()
            # self.train_data.re_random_item()
            for data in tqdm(train_loader):
                ind = data['index'].numpy()
                text = data['txt']  # type: torch.Tensor
                text = text.squeeze()
                if self.cuda:
                    text = text.cuda()
                hash_representation, label_predict = self.txt_model(text)
                self.txt_feature_buffer[ind] = hash_representation
                H = Variable(self.hash_buffer)
                logloss, hash_loss, label_loss = self.modality_loss(hash_representation, H, label_predict, ind)
                loss = logloss + hash_loss + label_loss
                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()
                self.remark_loss(logloss.item(), hash_loss.item(), label_loss.item(), loss.item())
            self.print_loss(epoch)
            self.plot_loss('text')
            self.reset_loss()
            # # gcn
            # for data in tqdm(train_loader):
            #     ind = data['index'].numpy()
            #     img_feature = self.img_feature_buffer[ind]
            #     txt_feature = self.txt_feature_buffer[ind]
            #     W = torch.norm(torch.matmul(img_feature, txt_feature.t()))
            #     feature = 0.5 * (img_feature * )
            #
            #     H = Variable(self.hash_buffer)
            #     logloss, hash_loss, label_loss = self.modality_loss(hash_representation, H, label_predict, ind)
            #     loss = logloss + hash_loss + label_loss
            #     self.optimizers[1].zero_grad()
            #     loss.backward()
            #     self.optimizers[1].step()
            #     self.remark_loss(logloss.item(), hash_loss.item(), label_loss.item(), loss.item())
            # self.print_loss(epoch)
            # self.plotter.plot('txt', 'dist loss', self.loss_store['dist loss'].avg)
            # self.reset_loss()

            self.valid(epoch)
            self.lr_schedule()
            self.plotter.next_epoch()
        print("train finish")

    def label_loss(self, hash_represent, hash_memory_bank, predict_label, ind):
        S = calc_neighbor(self.train_L[ind], self.train_L)
        theta = 1.0 / 2 * torch.matmul(hash_represent, hash_memory_bank.t())
        logloss = -torch.mean(S * theta - torch.log(1.0 + torch.exp(theta)))
        label_loss = torch.mean(torch.sum(torch.pow(self.train_L[ind] - predict_label, 2), dim=-1))
        logloss = logloss * self.parameters['alpha']
        label_loss = label_loss * self.parameters['beta']
        return logloss, label_loss

    def modality_loss(self, hash_represent, hash_memory_bank, predict_label, ind):
        S = calc_neighbor(self.train_L[ind], self.train_L)
        theta = 1.0 / 2 * torch.matmul(hash_represent, hash_memory_bank.t())
        logloss = -torch.mean(S * theta - torch.log(1.0 + torch.exp(theta)))
        hash_ground_truth = torch.sign(hash_memory_bank[ind])
        hash_loss = torch.mean(torch.sum(torch.pow(hash_ground_truth - hash_represent, 2), dim=-1))
        label_loss = torch.mean(torch.sum(torch.pow(self.train_L[ind, :] - predict_label, 2), dim=-1))
        logloss = logloss * self.parameters['alpha']
        hash_loss = hash_loss * self.parameters['beta']
        label_loss = label_loss * self.parameters['gamma']
        return logloss, hash_loss, label_loss


def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
    trainer = GCH(dataset_name, img_dir, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train()

