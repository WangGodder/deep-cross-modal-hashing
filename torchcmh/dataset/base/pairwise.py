# -*- coding: utf-8 -*-
# @Time    : 2019/6/27
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from .base import CrossModalTrainBase
import numpy as np
import torch


class CrossModalPairwiseTrain(CrossModalTrainBase):
    def __init__(self, img_dir: str, img_names: np.ndarray, txt_matrix: np.ndarray, label_matrix: np.ndarray, batch_size,
                 img_transform=None, txt_transform=None):
        super(CrossModalPairwiseTrain, self).__init__(img_dir, img_names, txt_matrix, label_matrix, img_transform, txt_transform)
        self.batch_size = batch_size
        self.ano_random_item = []
        self.re_random_item()

    def re_random_item(self):
        self.random_item = []
        self.ano_random_item = []
        for _ in range(self.length // self.batch_size):
            random_ind1 = np.random.permutation(range(self.length))
            # random_ind2 = np.random.permutation(range(self.train_num))
            self.random_item.append(random_ind1[:self.batch_size])
            self.ano_random_item.append(random_ind1[self.batch_size : self.batch_size * 2])

    def get_random_item(self, item):
        return self.random_item[item // self.batch_size][item % self.batch_size], self.ano_random_item[item // self.batch_size][
            item % self.batch_size]

    def __getitem__(self, item):
        item, ano_item = self.get_random_item(item)
        # ano_item = np.random.choice(np.setdiff1d(range(self.train_num), item), 1)[0]
        img = txt = None
        if self.img_read:
            img = self.read_img(item)
            ano_img = self.read_img(ano_item)
        if self.txt_read:
            txt = self.read_txt(item)
            ano_txt = self.read_txt(ano_item)
            # txt = torch.Tensor(self.txt[item][np.newaxis, :, np.newaxis])
            # ano_txt = torch.Tensor(self.txt[ano_item][np.newaxis, :, np.newaxis])
        label = torch.Tensor(self.label[item])
        ano_label = torch.Tensor(self.label[ano_item])
        index = torch.from_numpy(np.array(item))
        ano_index = torch.from_numpy(np.array(ano_item))
        back_dict = {'index': index, 'ano_index': ano_index,
                     'label': label, 'ano_label': ano_label}
        if self.img_read:
            back_dict['img'] = img
            back_dict['ano_img'] = ano_img
        if self.txt_read:
            back_dict['txt'] = txt
            back_dict['ano_txt'] = ano_txt
        return back_dict
        # if self.img_read is False:
        #     return {'index': index, 'ano_index': ano_index, 'txt': txt, 'ano_txt': ano_txt, 'label': label, 'ano_label': ano_label}
        # if self.txt_read is False:
        #     return {'index': index, 'ano_index': ano_index, 'img': img, 'ano_img': ano_img, 'label': label, 'ano_label': ano_label}
        # return {'index': index, 'ano_index': ano_index, 'txt': txt, 'ano_txt': ano_txt, 'img': img, 'ano_img': ano_img,
        #         'label': label, 'ano_label': ano_label}
