# -*- coding: utf-8 -*-
# @Time    : 2019/8/1
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import os
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchcmh.utils import get_plotter, AverageMeter
from torchcmh.utils import calc_map_k
from tqdm import tqdm


class TrainBase(object):
    def __init__(self, name, data_name, bit, batch_size, visdom=True, cuda=True):
        self.parameters = {}
        self.loss_store = []
        self.name = name
        self.bit = bit
        self.batch_size = batch_size
        self.cuda = cuda
        self.plotter = get_plotter(self.name) if visdom else None
        self.data_name = data_name
        self.img_model = self.txt_model = None
        self.train_data = self.valid_data = None
        self.save_code_freq = 10
        self.checkpoint_dir = os.path.join('..', 'checkpoints', self.name, data_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.max_epoch = 500
        self.lr = {}
        self.lr_decay = 0.9
        self.lr_decay_freq = 1
        self.optimizers = []
        self.schedulers = None
        self.max_mapi2t = self.max_mapt2i = 0.
        self.best_epoch = 0
        self.qB_img = self.qB_txt = self.rB_img = self.rB_txt = None
        self.best_train_img = self.best_train_txt = None

    def _init(self):
        self.schedulers = self.__lr_scheduler()
        print("training %s for %d bit. hyper-paramter list:" % (self.name, self.bit))
        for key, value in self.parameters.items():
            print("%8s: %4.3f" % (key, value))
        print("learning rate decay: %3.2f, decay frequency %2d, learning rate:" % (self.lr_decay, self.lr_decay_freq))
        for key, value in self.lr.items():
            print("%8s: %5.5f" % (key, value))
        print("img net: %s, txt net: %s" % (self.img_model.module_name, self.txt_model.module_name))
        loss_store = {}
        for loss_name in self.loss_store:
            loss_store[loss_name] = AverageMeter()
        self.loss_store = loss_store

    def __lr_scheduler(self):
        """
        create optimizers scheduler for all optimizers.
        default use StepLR scheduler base on lr_decay_freq and lr_decay
        :return:
        """
        schedulers = []
        for optimizer in self.optimizers:
            scheduler = StepLR(optimizer, self.lr_decay_freq, self.lr_decay)
            schedulers.append(scheduler)
        return schedulers

    def lr_schedule(self):
        """
        adjust the learning rate of all optimizers by schedulers.
        :return:
        """
        for scheduler in self.schedulers:
            scheduler.step()

    def plot_loss(self, title):
        if self.plotter:
            for name, loss in self.loss_store.items():
                self.plotter.plot(title, name, loss.avg)

    def print_loss(self, epoch):
        loss_str = "epoch: [%3d/%3d], " % (epoch+1, self.max_epoch)
        for name, value in self.loss_store.items():
            loss_str += name + " {:4.3f}".format(value.avg) + "\t"
        print(loss_str)

    def reset_loss(self):
        for store in self.loss_store.values():
            store.reset()

    def train(self, num_works=4):
        pass

    def object_function(self, *args):
        pass

    def remark_loss(self, *args):
        for i, loss_name in enumerate(self.loss_store.keys()):
            self.loss_store[loss_name].update(args[i].item())

    def load_model(self):
        """
        load models from default url.
        :return:
        """
        self.img_model.load_dict(os.path.join(self.checkpoint_dir, str(self.bit) + '-' + self.img_model.module_name + '.pth'))
        self.txt_model.load_dict(os.path.join(self.checkpoint_dir, str(self.bit) + '-' + self.txt_model.module_name + '.pth'))
        if self.cuda:
            self.img_model = self.img_model.cuda()
            self.txt_model = self.txt_model.cuda()

    @staticmethod
    def to_cuda(*args):
        """
        chagne all tensor from cpu tensor to cuda tensor
        :param args: tensor
        :return:
        """
        cuda_args = []
        for arg in args:
            cuda_args.append(arg.cuda())
        return cuda_args

    def valid(self, epoch):
        """
        valid current training model, and save the best model and hash code.
        :param epoch: current epoch
        :return:
        """
        mapi2t, mapt2i,  qB_img, qB_txt, rB_img, rB_txt = \
            self.valid_calc(self.img_model, self.txt_model, self.valid_data, self.bit, self.batch_size, return_hash=True)
        if mapt2i + mapi2t >= self.max_mapi2t + self.max_mapt2i:
            self.max_mapi2t = mapi2t
            self.max_mapt2i = mapt2i
            self.best_epoch = epoch
            self.img_model.save_dict(os.path.join(self.checkpoint_dir, str(self.bit) + '-' + self.img_model.module_name + '.pth'))
            self.txt_model.save_dict(os.path.join(self.checkpoint_dir, str(self.bit) + '-' + self.txt_model.module_name + '.pth'))
            self.qB_img = qB_img.cpu()
            self.qB_txt = qB_txt.cpu()
            self.rB_img = rB_img.cpu()
            self.rB_txt = rB_txt.cpu()
            # self.best_train_img, self.best_train_txt = self.get_train_hash()
        print('epoch: [%3d/%3d], valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f, max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f in epoch %d' %
              (epoch + 1, self.max_epoch, mapi2t, mapt2i, self.max_mapi2t, self.max_mapt2i, self.best_epoch + 1))
        if self.plotter:
            self.plotter.plot("mAP", 'i->t', mapi2t.item())
            self.plotter.plot("mAP", "t->i", mapt2i.item())
        self.save_code(epoch)

    @staticmethod
    def valid_calc(img_model, txt_model, dataset, bit, batch_size, drop_integer=False, return_hash=False):
        """
        get valid set hash code and calculate mAP
        :param img_model:
        :param txt_model:
        :param dataset:
        :param bit:
        :param batch_size:
        :param drop_integer:
        :param return_hash:
        :return:
        """
        # get query img and txt binary code
        dataset.query()
        qB_img = TrainBase.get_img_code(img_model, dataset, bit, batch_size, drop_integer)
        qB_txt = TrainBase.get_txt_code(txt_model, dataset, bit, batch_size, drop_integer)
        query_label = dataset.get_all_label()
        # get retrieval img and txt binary code
        dataset.retrieval()
        rB_img = TrainBase.get_img_code(img_model, dataset, bit, batch_size, drop_integer)
        rB_txt = TrainBase.get_txt_code(txt_model, dataset, bit, batch_size, drop_integer)
        retrieval_label = dataset.get_all_label()
        mAPi2t = calc_map_k(qB_img, rB_txt, query_label, retrieval_label)
        mAPt2i = calc_map_k(qB_txt, rB_img, query_label, retrieval_label)
        if return_hash:
            return mAPi2t, mAPt2i, qB_img.cpu(), qB_txt.cpu(), rB_img.cpu(), rB_txt.cpu()
        return mAPi2t, mAPt2i

    def get_train_hash(self):
        img_hash = self.get_img_code(self.img_model, self.train_data, self.bit, self.batch_size)
        txt_hash = self.get_txt_code(self.txt_model, self.train_data, self.bit, self.batch_size)
        return img_hash, txt_hash

    @staticmethod
    def calc_map(qB_img, qB_txt, rB_img, rB_txt, query_label, retrieval_label):
        mAPi2t = calc_map_k(qB_img, rB_txt, query_label, retrieval_label)
        mAPt2i = calc_map_k(qB_txt, rB_img, query_label, retrieval_label)
        return mAPi2t, mAPt2i

    @staticmethod
    def get_img_code(img_model, dataset, bit, batch_size, drop_integer=False, cuda=True):
        dataset.img_load()
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=drop_integer)
        B_img = torch.zeros(len(dataset), bit, dtype=torch.float)
        if cuda:
            B_img = B_img.cuda()
        img_model.eval()
        for data in tqdm(dataloader):
            index = data['index'].numpy()  # type: np.ndarray
            img = data['img']  # type: torch.Tensor
            if cuda:
                img = img.cuda()

            f = img_model(img)
            B_img[index, :] = f.data
        return B_img

    @staticmethod
    def get_txt_code(txt_model, dataset, bit, batch_size, drop_integer=False, cuda=True):
        dataset.txt_load()
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, pin_memory=True, drop_last=drop_integer)
        B_txt = torch.zeros(len(dataset), bit, dtype=torch.float)
        if cuda:
            B_txt = B_txt.cuda()
        txt_model.eval()
        for data in tqdm(dataloader):
            index = data['index'].numpy()  # type: np.ndarray
            txt = data['txt']  # type: torch.Tensor
            if cuda:
                txt = txt.cuda()

            g = txt_model(txt)
            B_txt[index, :] = g.data
        return B_txt

    def save_code(self, epoch):
        """
        save valid hash code in default url.
        :param epoch: current epoch
        :return:
        """
        if (epoch+1) % self.save_code_freq != 0 or epoch - self.best_epoch < 10 is None:
            return
        qB_img = torch.sign(self.qB_img).numpy().astype(int)
        qB_txt = torch.sign(self.qB_txt).numpy().astype(int)
        rB_img = torch.sign(self.rB_img).numpy().astype(int)
        rB_txt = torch.sign(self.rB_txt).numpy().astype(int)
        import scipy.io as sio
        path = os.path.join(self.checkpoint_dir, str(self.bit) + '-hash.mat')
        sio.savemat(path, {"q_img": qB_img, "q_txt": qB_txt, "r_img": rB_img, "r_txt": rB_txt})
        print("save hash code in %s" % path)

    def resume_train(self, state_path):
        pass
