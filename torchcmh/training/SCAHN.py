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
from torchcmh.models.ASCHN import resnet18, resnet34, get_MS_Text
from torchcmh.training.base import TrainBase
from torchcmh.utils import calc_neighbor
from torchcmh.dataset.utils import pairwise_data

__all__ = ['train']


class SCAHN(TrainBase):
    def __init__(self, data_name: str, img_dir: str, bit: int, img_net, visdom=True, batch_size=128, cuda=True, **kwargs):
        super(SCAHN, self).__init__("SCAHN", data_name, bit, batch_size, visdom, cuda)
        self.train_data, self.valid_data = pairwise_data(data_name, img_dir, batch_size=batch_size)
        self.loss_store = ['inter loss', 'intra loss', 'pairwise intra loss', 'quantization loss', 'loss']
        self.parameters = {'fusion num': 4, 'beta': 2 ** np.log2(bit / 32), 'lambda': 1, 'gamma': 1, 'eta': 1/bit}
        self.lr = {'img': 10**(-1.1), 'txt': 10**(-1.1)}
        self.lr_decay_freq = 1
        self.lr_decay = 0.98

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
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, drop_last=True, num_workers=num_works, shuffle=False,
                                  pin_memory=True)
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

                inter_loss, intra_loss, intra_pair_loss, quantization_loss = self.object_function(hash1_layers, hash2_layers, hash1, hash2, sample_L1, sample_L2, F, G,ind1, ind2)

                loss = inter_loss + intra_pair_loss + intra_loss + quantization_loss

                self.optimizers[0].zero_grad()
                loss.backward()
                self.optimizers[0].step()
                self.remark_loss(inter_loss, intra_loss, intra_pair_loss, quantization_loss, loss)
                # self.loss_store['inter loss'].update(inter_loss.item())
                # self.loss_store['intra loss'].update(intra_loss.item())
                # self.loss_store['pairwise intra loss'].update(intra_pair_loss.item())
                # self.loss_store['quantization loss'].update(quantization_loss.item())
                # self.loss_store['loss'].update(loss.item())
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

                inter_loss, intra_loss, intra_pair_loss, quantization_loss = self.object_function(hash1_layers, hash2_layers, hash1, hash2,
                                                                                                  sample_L1, sample_L2, G, F, ind1, ind2)

                loss = inter_loss + intra_pair_loss + intra_loss + quantization_loss

                self.optimizers[1].zero_grad()
                loss.backward()
                self.optimizers[1].step()
                self.remark_loss(inter_loss, intra_loss, intra_pair_loss, quantization_loss, loss)
                # self.loss_store['inter loss'].update(inter_loss.item())
                # self.loss_store['intra loss'].update(intra_loss.item())
                # self.loss_store['pairwise intra loss'].update(intra_pair_loss.item())
                # self.loss_store['quantization loss'].update(quantization_loss.item())
                # self.loss_store['loss'].update(loss.item())
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
        inter_loss1, inter_loss2 = calc_inter_loss(hash_layers1, hash_layer2, S_inter1, S_inter2, G, self.parameters['beta'])
        inter_loss = 0.5 * (inter_loss1 + inter_loss2)
        intra_loss1, intra_loss2 = calc_intra_loss(hash_layers1, hash_layer2, S_inter1, S_inter2, F, self.parameters['beta'])
        intra_loss = 0.5 * (intra_loss1 + intra_loss2) * self.parameters['lambda']
        intra_pair_loss = calc_intra_pairwise_loss(hash_layers1, hash_layer2, S_intra, self.parameters['beta'])
        intra_pair_loss = intra_pair_loss * self.parameters['gamma']
        quantization_loss1 = torch.mean(torch.sum(torch.pow(self.B[ind1, :] - final_hash1, 2), dim=1))
        quantization_loss2 = torch.mean(torch.sum(torch.pow(self.B[ind2, :] - final_hash2, 2), dim=1))
        quantization_loss = 0.5 * (quantization_loss1 + quantization_loss2) * self.parameters['eta']
        return inter_loss, intra_loss, intra_pair_loss, quantization_loss


def train(dataset_name: str, img_dir: str, bit: int, img_net_name='resnet34', visdom=True, batch_size=128, cuda=True, **kwargs):
    img_net = resnet34 if img_net_name == 'resnet34' else resnet18
    trainer = SCAHN(dataset_name, img_dir, img_net, bit, visdom, batch_size, cuda, **kwargs)
    trainer.train()


# def train(dataset_name: str, img_dir: str, bit: int, img_net_name='resnet34', visdom=True, batch_size=128, cuda=True):
#     lr = 10 ** (-1.1)
#     lr_end = 10 ** (-6.5)
#     max_epoch = 500
#     alpha = 2 ** np.log2(bit / 32)
#     fusion_num = 2
#     lambdaa = 1
#     gamma1 = 1
#     gamma2 = 1
#     eta = 2
#     print("training %s, for %3d bit. hyper-paramter list:\n lambda = %3.2f \n gamma = %3.2f \n eta = %3.2f\n fusion num: %d" %
#           (name, bit, lambdaa, gamma1, eta, fusion_num))
#     checkpoint_dir = os.path.join('..', 'checkpoints', name)
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#
#     if visdom:
#         plotter = get_plotter(name)
#
#     train_data, valid_data = pairwise_data(dataset_name, img_dir, batch_size=batch_size)
#     train_num = len(train_data)
#
#     if img_net_name == 'resnet34':
#         img_model = resnet34(bit, fusion_num=fusion_num)
#     elif img_net_name == 'resnet18':
#         img_model = resnet18(bit, fusion_num=fusion_num)
#     else:
#         raise ValueError("wrong image model name %s" % img_net_name)
#     txt_model = get_MS_Text(train_data.get_tag_length(), bit, fusion_num)
#
#     train_L = train_data.get_all_label()
#     F_buffer = torch.randn(len(train_data), bit)
#     G_buffer = torch.randn(len(train_data), bit)
#
#     if cuda:
#         train_L = train_L.cuda()
#         F_buffer = F_buffer.cuda()
#         G_buffer = G_buffer.cuda()
#         img_model = img_model.cuda()
#         txt_model = txt_model.cuda()
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
#     best_epoch = 0
#     single_max_i2t = single_max_t2i = 0.
#     single_max_i2t_epoch = single_max_t2i_epoch = 0
#     train_loader = DataLoader(train_data, batch_size=batch_size, drop_last=True, num_workers=4, shuffle=False, pin_memory=True)
#     for epoch in range(max_epoch):
#         img_model.train()
#         txt_model.train()
#         train_data.img_load()
#         train_data.re_random_item()
#         for data in tqdm(train_loader):
#             ind1 = data['index'].numpy()
#             ind2 = data['ano_index'].numpy()
#
#             sample_L1 = data['label']  # type: torch.Tensor
#             sample_L2 = data['ano_label']  # type: torch.Tensor
#             image1 = data['img']  # type: torch.Tensor
#             image2 = data['ano_img']  # type: torch.Tensor
#             if cuda:
#                 image1 = image1.cuda()
#                 image2 = image2.cuda()
#                 sample_L1 = sample_L1.cuda()
#                 sample_L2 = sample_L2.cuda()
#
#             # get similarity matrix
#             S_inter1 = calc_neighbor(sample_L1, train_L)
#             S_inter2 = calc_neighbor(sample_L2, train_L)
#             S_intra = calc_neighbor(sample_L1, sample_L2)
#
#             # get output from img net
#             middle_hash1, hash1 = img_model(image1)
#             middle_hash2, hash2 = img_model(image2)
#
#             hash1_layers = middle_hash1
#             hash2_layers = middle_hash2
#
#             hash1_layers.append(hash1)
#             hash2_layers.append(hash2)
#
#             F_buffer[ind1, :] = hash1.data
#             F_buffer[ind2, :] = hash2.data
#             F = Variable(F_buffer)
#             G = Variable(G_buffer)
#
#
#
#             loss = inter_loss + intra_pair_loss + intra_loss + quantization_loss
#
#             optimizer_img.zero_grad()
#             loss.backward()
#             optimizer_img.step()
#
#             # store loss
#             inter_loss_store.update(inter_loss.item())
#             intra_loss_store.update(intra_loss.item())
#             intra_pairwise_loss_store.update(intra_pair_loss.item())
#             quantization_loss_store.update(quantization_loss.item())
#             loss_store.update(loss.item())
#         print("inter loss: %4.3f, intra loss: %4.3f, intra pair loss: %4.3f, quan loss: %4.3f, balance loss: %4.3f, loss: %4.3f" %
#               (inter_loss_store.avg, intra_loss_store.avg, intra_pairwise_loss_store.avg, quantization_loss_store.avg,
#                balance_loss_store.avg, loss_store.avg))
#         if plotter is not None:
#             plotter.plot('img_loss', 'inter loss', inter_loss_store.avg)
#             plotter.plot('img_loss', 'intra loss', intra_loss_store.avg)
#             plotter.plot('img_loss', 'intra pair loss', intra_pairwise_loss_store.avg)
#             plotter.plot('img_loss', 'quantization', quantization_loss_store.avg)
#             plotter.plot('img_loss', 'loss', loss_store.avg)
#         weight = img_model.weight.weight  # type: torch.Tensor
#         weight = torch.mean(weight, dim=1)
#         for i in range(weight.shape[0]):
#             plotter.plot("img ms weight", 'part' + str(i), weight[i].item())
#
#             # reset loss store
#         inter_loss_store.reset()
#         intra_loss_store.reset()
#         intra_pairwise_loss_store.reset()
#         quantization_loss_store.reset()
#         balance_loss_store.reset()
#         loss_store.reset()
#
#         # training text net
#         train_data.txt_load()
#         train_data.re_random_item()
#         for data in tqdm(train_loader):
#             ind1 = data['index'].numpy()
#             ind2 = data['ano_index'].numpy()
#
#             sample_L1 = data['label']  # type: torch.Tensor
#             sample_L2 = data['ano_label']  # type: torch.Tensor
#             tag1 = data['txt']  # type: torch.Tensor
#             tag2 = data['ano_txt']  # type: torch.Tensor
#             if cuda:
#                 tag1 = tag1.cuda()
#                 tag2 = tag2.cuda()
#                 sample_L1 = sample_L1.cuda()
#                 sample_L2 = sample_L2.cuda()
#
#             # get similarity matrix
#             S_inter1 = calc_neighbor(sample_L1, train_L)
#             S_inter2 = calc_neighbor(sample_L2, train_L)
#             S_intra = calc_neighbor(sample_L1, sample_L2)
#
#             # get output from img net
#             middle_hash1, hash1 = txt_model(tag1)
#             middle_hash2, hash2 = txt_model(tag2)
#
#             hash1_layers = middle_hash1
#             hash2_layers = middle_hash2
#
#             hash1_layers.append(hash1)
#             hash2_layers.append(hash2)
#
#             # input hash from net into all image hash matrix
#             G_buffer[ind1, :] = hash1.data
#             G_buffer[ind2, :] = hash2.data
#             F = Variable(F_buffer)
#             G = Variable(G_buffer)
#
#             # calculate the inter loss
#             inter_loss1, inter_loss2 = calc_inter_loss(hash1_layers, hash2_layers, S_inter1, S_inter2, F, alpha)
#             inter_loss = 0.5 * (inter_loss1 + inter_loss2) / (batch_size * train_num)
#
#             # calculate the intra loss
#             intra_loss1, intra_loss2 = calc_intra_loss(hash1_layers, hash2_layers, S_inter1, S_inter2, G, alpha)
#             intra_loss = (intra_loss1 + intra_loss2) * 0.5
#             intra_loss /= (batch_size * train_num)
#             intra_loss *= lambdaa
#
#             # calculate the intra pair loss
#             intra_pair_loss = calc_intra_pairwise_loss(hash1_layers, hash2_layers, S_intra, alpha)
#             intra_pair_loss /= (batch_size * batch_size)
#             intra_pair_loss *= gamma2
#
#             # calculate the quantization loss
#             quantization_loss1 = torch.sum(torch.pow(B[ind1, :] - hash1, 2))
#             quantization_loss2 = torch.sum(torch.pow(B[ind2, :] - hash2, 2))
#             quantization_loss = 0.5 * (quantization_loss1 + quantization_loss2) / (batch_size * bit)
#             quantization_loss *= eta
#
#             loss = inter_loss + intra_pair_loss + quantization_loss + intra_loss
#
#             # bp
#             optimizer_txt.zero_grad()
#             loss.backward()
#             optimizer_txt.step()
#
#             # store loss
#             inter_loss_store.update(inter_loss.item())
#             intra_pairwise_loss_store.update(intra_pair_loss.item())
#             intra_loss_store.update(intra_loss.item())
#             quantization_loss_store.update(quantization_loss.item())
#             loss_store.update(loss.item())
#         print("inter loss: %4.3f, intra loss: %4.3f, intra pair loss: %4.3f,  quan loss: %4.3f, balance loss: %4.3f, loss: %4.3f" %
#               (inter_loss_store.avg, intra_loss_store.avg, intra_pairwise_loss_store.avg, quantization_loss_store.avg,
#                balance_loss_store.avg, loss_store.avg))
#         if plotter is not None:
#             plotter.plot('txt_loss', 'inter loss', inter_loss_store.avg)
#             plotter.plot('txt_loss', 'intra loss', intra_loss_store.avg)
#             plotter.plot('txt_loss', 'intra pair loss', intra_pairwise_loss_store.avg)
#             plotter.plot('txt_loss', 'quantization', quantization_loss_store.avg)
#             plotter.plot('txt_loss', 'loss', loss_store.avg)
#
#         weight = txt_model.weight.weight  # type: torch.Tensor
#         weight = torch.mean(weight, dim=1)
#         for i in range(weight.shape[0]):
#             plotter.plot("txt ms weight", 'part' + str(i), weight[i].item())
#
#         # reset loss store
#         inter_loss_store.reset()
#         intra_loss_store.reset()
#         intra_pairwise_loss_store.reset()
#         quantization_loss_store.reset()
#         balance_loss_store.reset()
#         loss_store.reset()
#
#         # update B
#         B = torch.sign(F_buffer + G_buffer)
#         loss = calc_loss(B, F, G, Sim, gamma1, gamma2, eta, alpha)
#         print('...epoch: %3d, loss: %4.3f, lr: %f' % (epoch + 1, loss, lr))
#         plotter.plot("object loss", 'loss', loss.item())
#
#         mapi2t, mapt2i, qB_img, qB_txt, rB_img, rB_txt = valid(img_model, txt_model, valid_data, bit, batch_size, return_hash=True)
#         if single_max_i2t <= mapi2t:
#             single_max_i2t = mapi2t
#             single_max_i2t_epoch = epoch
#         if single_max_t2i <= mapt2i:
#             single_max_t2i = mapt2i
#             single_max_t2i_epoch = epoch
#         if mapt2i + mapi2t >= max_mapi2t + max_mapt2i:
#             img_model.save_entire(os.path.join(checkpoint_dir, str(bit) + '-' + img_model.module_name + '.pth'))
#             txt_model.save_entire(os.path.join(checkpoint_dir, str(bit) + '-' + txt_model.module_name + '.pth'))
#             max_mapi2t = mapi2t
#             max_mapt2i = mapt2i
#             best_epoch = epoch
#             if bit > 128:
#                 bit_scalable(img_model, txt_model, qB_img, qB_txt, rB_img, rB_txt, valid_data)
#
#         print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' %
#               (epoch + 1, mapi2t, mapt2i, max_mapi2t, max_mapt2i))
#         print('double best epoch: %3d, i->t best: %3.4f, on epoch: %3d, t->i best: %3.4f, on epoch: %3d' %
#               (best_epoch + 1, single_max_i2t, single_max_i2t_epoch + 1, single_max_t2i, single_max_t2i_epoch + 1))
#         plotter.plot("mAP", 'i->t', mapi2t.item())
#         plotter.plot("mAP", "t->i", mapt2i.item())
#
#         plotter.next_epoch()
#         lr = learning_rate[epoch + 1]
#
#         for param in optimizer_img.param_groups:
#             param['lr'] = lr
#         for param in optimizer_txt.param_groups:
#             param['lr'] = lr


def calc_inter_loss(hash1_layers, hash2_layers, S1, S2, O, alpha):
    inter_loss1 = 0
    for index, hash1_layer in enumerate(hash1_layers):
        theta = 1.0 / alpha * torch.matmul(hash1_layer, O.t())
        logloss = -torch.mean(S1 * theta - torch.log(1 + torch.exp(theta)))
        if torch.isinf(logloss):
            print("the log loss is inf in hash1 of layer %d, with the max of theta is %3.4f" % (index, torch.max(theta).data))
        inter_loss1 += logloss
    inter_loss2 = 0
    for index, hash2_layer in enumerate(hash2_layers):
        theta = 1.0 / alpha * torch.matmul(hash2_layer, O.t())
        logloss = -torch.mean(S2 * theta - torch.log(1 + torch.exp(theta)))
        if torch.isinf(logloss):
            print("the log loss is inf in hash2 of layer %d, with the max of theta is %3.4f" % (index, torch.max(theta).data))
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
            print("the log loss is inf in hash1 and hash2 of layer %d, with the max of theta is %3.4f" % (index, torch.max(theta).data))
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
