# -*- coding: utf-8 -*-
# @Time    : 2019/7/14
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import numpy as np
import torch
from torch.optim import SGD
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchcmh.models import cnnf, MLP
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor
from torchcmh.dataset import single_data


class PRDH(TrainBase):
    """
    Yang et al. Pairwise relationship guided deep hashing for cross-modal retrieval.
    In Thirty-First AAAI Conference on Artificial Intelligence.2017
    """
    def __init__(self, data_name, img_dir, bit, visdom=True, batch_size=128, cuda=True, **kwargs):
        super(PRDH, self).__init__("PRDH", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = single_data(data_name, img_dir, batch_size=batch_size, **kwargs)
        self.loss_store = ["inter loss", 'intra loss', 'decorrelation loss', 'regularization loss', 'loss']
        self.parameters = {'gamma': 1, 'lambda': 1}
        self.max_epoch = 500
        self.lr = {'img': 10 ** (-1.5), 'txt': 10 ** (-1.5)}
        self.lr_decay_freq = 1
        self.lr_decay = (10 ** (-1.5) - 1e-6) / self.max_epoch
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

                inter_loss, intra_loss, decorrelation_loss, reg_loss = self.object_function(cur_f, sample_L, G, F, ind)
                loss = intra_loss + inter_loss + reg_loss
                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()

                self.remark_loss(intra_loss, intra_loss, decorrelation_loss, reg_loss, loss)
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

                inter_loss, intra_loss, decorrelation_loss, reg_loss = self.object_function(cur_g, sample_L, F, G, ind)
                loss = intra_loss + inter_loss + decorrelation_loss + reg_loss
                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()
                self.remark_loss(intra_loss, intra_loss, decorrelation_loss, reg_loss, loss)

            self.print_loss(epoch)
            self.plot_loss("txt loss")
            self.reset_loss()
            self.B = torch.sign(self.F_buffer + self.G_buffer)
            self.valid(epoch)
            self.lr_schedule()
            self.plotter.next_epoch()
        print("train finish")

    def object_function(self, cur_h: torch.Tensor, sample_label: torch.Tensor, A: torch.Tensor, C: torch.Tensor, ind):
        unupdated_ind = np.setdiff1d(range(self.num_train), ind)
        S = calc_neighbor(sample_label, self.train_L)
        theta = 1.0 / 2 * torch.matmul(cur_h, A.t())
        inter_loss = -torch.mean(S * theta - torch.log(1.0 + torch.exp(theta)))

        theta = 1.0 / 2 * torch.matmul(cur_h, C.t())
        intra_loss = -torch.mean(S * theta - torch.log(1.0 + torch.exp(theta)))

        decorrelation_loss = calc_decorrelation_loss(cur_h)
        decorrelation_loss *= self.parameters['lambda']

        quantization = torch.sum(torch.pow(self.B - C, 2))
        quantization /= (self.num_train * self.batch_size)
        balance = torch.sum(torch.pow(cur_h.t().mm(self.ones) + self.C[unupdated_ind].t().mm(self.ones_), 2))
        balance /= (self.num_train * self.batch_size)
        regularization_loss = quantization + balance
        regularization_loss *= self.parameters['gamma']

        return inter_loss, intra_loss, decorrelation_loss, regularization_loss

# def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True):
#     lr = 10 ** (-1.5)
#     lr_end = 10 ** (-6.0)
#     max_epoch = 500
#     lambdaa = 1
#     gamma = 1
#     print("training %s, for %3d bit. hyper-paramter list:\n gamma = %3.2f \n lambdaa = %3.2f" % (name, bit, gamma, lambdaa))
#     checkpoint_dir = os.path.join('..', 'checkpoints', name, dataset_name)
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#
#     if visdom:
#         plotter = get_plotter(name)
#
#     train_data, valid_data = single_data(dataset_name, img_dir, batch_size=batch_size)
#
#     img_model = vgg_f.get_vgg_f(bit)
#     txt_model = mlp.MLP(train_data.get_tag_length(), bit)
#     num_train = len(train_data)
#
#     train_L = train_data.get_all_label()
#     F_buffer = torch.randn(num_train, bit)
#     G_buffer = torch.randn(num_train, bit)
#     ones = torch.ones(batch_size, 1)
#     ones_ = torch.ones(num_train - batch_size, 1)
#
#     if cuda:
#         img_model = img_model.cuda()
#         txt_model = txt_model.cuda()
#         train_L = train_L.cuda()
#         F_buffer = F_buffer.cuda()
#         G_buffer = G_buffer.cuda()
#         ones = ones.cuda()
#         ones_ = ones_.cuda()
#
#     Sim = calc_neighbor(train_L, train_L)
#     B = torch.sign(F_buffer + G_buffer)
#
#     optimizer_img = SGD(img_model.parameters(), lr=lr)
#     optimizer_txt = SGD(txt_model.parameters(), lr=lr)
#
#     learning_rate = np.linspace(lr, lr_end, max_epoch + 1)
#
#     max_mapi2t = max_mapt2i = 0.
#     train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=4, shuffle=False, pin_memory=True)
#     for epoch in range(max_epoch):
#         img_model.train()
#         txt_model.train()
#         train_data.img_load()
#         train_data.re_random_item()
#         for data in tqdm(train_loader):
#             ind = data['index'].numpy()
#             unupdated_ind = np.setdiff1d(range(num_train), ind)
#
#             sample_L = data['label']  # type: torch.Tensor
#             image = data['img']  # type: torch.Tensor
#             if cuda:
#                 image = image.cuda()
#                 sample_L = sample_L.cuda()
#
#             # get similarity matrix
#             S = calc_neighbor(sample_L, train_L)
#             hash = img_model(image)
#             hash = torch.tanh(hash)
#
#             F_buffer[ind, :] = hash.data
#             F = Variable(F_buffer)
#             G = Variable(G_buffer)
#
#             theta = 1.0 / 2 * torch.matmul(hash, G.t())
#             inter_loss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta)))
#             inter_loss /= (batch_size * num_train)
#
#             theta = 1.0 / 2 * torch.matmul(hash, F.t())
#             intra_loss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta)))
#             intra_loss /= (batch_size * num_train)
#
#             decorrelation_loss = calc_decorrelation_loss(hash)
#             decorrelation_loss *= lambdaa
#
#             quantization = torch.sum(torch.pow(B - F, 2))
#             quantization /= (num_train * batch_size)
#             balance = torch.sum(torch.pow(hash.t().mm(ones) + F[unupdated_ind].t().mm(ones_), 2))
#             balance /= (num_train * batch_size)
#             regularization_loss = quantization + balance
#             regularization_loss *= gamma
#
#             loss = inter_loss + intra_loss + decorrelation_loss + regularization_loss
#
#             optimizer_img.zero_grad()
#             loss.backward()
#             optimizer_img.step()
#
#             inter_loss_store.update(inter_loss.item())
#             intra_loss_store.update(intra_loss.item())
#             decorrelation_loss_store.update(decorrelation_loss.item())
#             regularization_loss_store.update(regularization_loss.item())
#             loss_store.update(loss.item())
#         print("loss: %4.4f, inter loss: %4.4f, intra loss: %4.4f, decorrelation loss: %4.4f, regularzation: %4.4f" %
#               (loss_store.avg, inter_loss_store.avg, intra_loss_store.avg, decorrelation_loss_store.avg, regularization_loss_store.avg))
#         if plotter is not None:
#             plotter.plot("img loss", "loss", loss_store.avg)
#             plotter.plot("img loss", "inter loss", inter_loss_store.avg)
#             plotter.plot("img loss", "intra loss", intra_loss_store.avg)
#             plotter.plot("img loss", "decorrelation loss", decorrelation_loss_store.avg)
#             plotter.plot("img loss", "regularzation loss", regularization_loss_store.avg)
#         inter_loss_store.reset()
#         intra_loss_store.reset()
#         decorrelation_loss_store.reset()
#         regularization_loss_store.reset()
#         loss_store.reset()
#
#         train_data.txt_load()
#         train_data.re_random_item()
#         for data in tqdm(train_loader):
#             ind = data['index'].numpy()
#             unupdated_ind = np.setdiff1d(range(num_train), ind)
#
#             sample_L = data['label']  # type: torch.Tensor
#             txt = data['txt']  # type: torch.Tensor
#             if cuda:
#                 txt = txt.cuda()
#                 sample_L = sample_L.cuda()
#
#             # get similarity matrix
#             S = calc_neighbor(sample_L, train_L)
#             hash = txt_model(txt)
#             hash = torch.tanh(hash)
#
#             F_buffer[ind, :] = hash.data
#             F = Variable(F_buffer)
#             G = Variable(G_buffer)
#
#             theta = 1.0 / 2 * torch.matmul(hash, F.t())
#             inter_loss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta)))
#             inter_loss /= (batch_size * num_train)
#
#             theta = 1.0 / 2 * torch.matmul(hash, G.t())
#             intra_loss = -torch.sum(S * theta - torch.log(1.0 + torch.exp(theta)))
#             intra_loss /= (batch_size * num_train)
#
#             decorrelation_loss = calc_decorrelation_loss(hash)
#             decorrelation_loss *= lambdaa
#
#             quantization = torch.sum(torch.pow(B - G, 2))
#             quantization /= (num_train * batch_size)
#             balance = torch.sum(torch.pow(hash.t().mm(ones) + G[unupdated_ind].t().mm(ones_), 2))
#             balance /= (num_train * batch_size)
#             regularization_loss = quantization + balance
#             regularization_loss *= gamma
#
#             loss = inter_loss + intra_loss + decorrelation_loss + regularization_loss
#
#             optimizer_txt.zero_grad()
#             loss.backward()
#             optimizer_txt.step()
#
#             inter_loss_store.update(inter_loss.item())
#             intra_loss_store.update(intra_loss.item())
#             decorrelation_loss_store.update(decorrelation_loss.item())
#             regularization_loss_store.update(regularization_loss.item())
#             loss_store.update(loss.item())
#         print("loss: %4.4f, inter loss: %4.4f, intra loss: %4.4f, decorrelation loss: %4.4f, regularzation: %4.4f" %
#               (loss_store.avg, inter_loss_store.avg, intra_loss_store.avg, decorrelation_loss_store.avg, regularization_loss_store.avg))
#         if plotter is not None:
#             plotter.plot("txt loss", "loss", loss_store.avg)
#             plotter.plot("txt loss", "inter loss", inter_loss_store.avg)
#             plotter.plot("txt loss", "intra loss", intra_loss_store.avg)
#             plotter.plot("txt loss", "decorrelation loss", decorrelation_loss_store.avg)
#             plotter.plot("txt loss", "regularzation loss", regularization_loss_store.avg)
#         inter_loss_store.reset()
#         intra_loss_store.reset()
#         decorrelation_loss_store.reset()
#         regularization_loss_store.reset()
#         loss_store.reset()
#
#         # update B
#         B = torch.sign(F_buffer + G_buffer)
#         loss = calc_loss(B, F, G, Variable(Sim), lambdaa, gamma)
#         if plotter is not None:
#             plotter.plot("total loss", "loss", loss.item())
#         print('...epoch: %3d, loss: %3.3f, lr: %f' % (epoch + 1, loss.data, lr))
#
#         mapi2t, mapt2i = valid(img_model, txt_model, valid_data, bit, batch_size)
#         if mapt2i + mapi2t >= max_mapi2t + max_mapt2i:
#             max_mapi2t = mapi2t
#             max_mapt2i = mapt2i
#             img_model.save_entire(os.path.join(checkpoint_dir, str(bit) + '-' + img_model.module_name + '.pth'))
#             txt_model.save_entire(os.path.join(checkpoint_dir, str(bit) + '-' + txt_model.module_name + '.pth'))
#         print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' %
#               (epoch + 1, mapi2t, mapt2i, max_mapi2t, max_mapt2i))
#         if plotter is not None:
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


def train(dataset_name: str, img_dir: str, bit: int, visdom=True, batch_size=128, cuda=True, **kwargs):
    trainer = PRDH(dataset_name, img_dir, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train()


def calc_decorrelation_loss(hash_code: torch.Tensor):
    miu = torch.mean(hash_code, dim=0, keepdim=True)
    C = torch.matmul(torch.mean(hash_code - miu, dim=0, keepdim=True).t(), torch.mean(hash_code - miu, dim=0, keepdim=True))
    decorrelation_loss = torch.sum(torch.pow(C, 2)) - torch.sum(torch.pow(torch.diag(C), 2))
    return decorrelation_loss * 0.5


def calc_loss(B, F, G, Sim, lambdaa, gamma):
    theta = torch.matmul(F, G.transpose(0, 1)) / 2
    inter_loss = torch.sum(torch.log(1 + torch.exp(theta)) - Sim * theta)
    theta_f = torch.matmul(F, F.t()) / 2
    theta_g = torch.matmul(G, G.t()) / 2
    intra_loss_f = torch.sum(torch.log(1 + torch.exp(theta_f) - Sim * theta_f))
    intra_loss_g = torch.sum(torch.log(1 + torch.exp(theta_g) - Sim * theta_g))
    intra_loss = intra_loss_f + intra_loss_g
    miu_f = torch.mean(F, dim=0, keepdim=True)
    C_f = torch.matmul(torch.sum(F - miu_f, dim=0, keepdim=True).t(), torch.sum(F - miu_f, dim=0, keepdim=True))
    decorrelation_f = torch.sum(torch.pow(C_f, 2)) - torch.sum(torch.pow(torch.diag(C_f), 2))
    decorrelation_f *= 0.5
    miu_g = torch.mean(G, dim=0, keepdim=True)
    C_g = torch.matmul(torch.sum(G - miu_g, dim=0, keepdim=True).t(), torch.sum(G - miu_g, dim=0, keepdim=True))
    decorrelation_g = torch.sum(torch.pow(C_g, 2)) - torch.sum(torch.pow(torch.diag(C_g), 2))
    decorrelation_g *= 0.5
    decorrelation_loss = decorrelation_f + decorrelation_g
    quantization = torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2))
    balance = torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))
    regularization_loss = quantization + balance
    loss = inter_loss + intra_loss + lambdaa * decorrelation_loss + gamma * regularization_loss
    return loss

