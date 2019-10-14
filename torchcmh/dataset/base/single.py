# -*- coding: utf-8 -*-
# @Time    : 2019/6/27
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from .base import CrossModalTrainBase
import numpy as np
import torch


class CrossModalSingleTrain(CrossModalTrainBase):
    def __init__(self, img_dir: str, img_names: np.ndarray, txt_matrix: np.ndarray, label_matrix: np.ndarray, batch_size,
                 img_transform=None, txt_transform=None):
        super(CrossModalSingleTrain, self).__init__(img_dir, img_names, txt_matrix, label_matrix, img_transform, txt_transform)
        self.batch_size = batch_size
        self.re_random_item()

    def re_random_item(self):
        self.random_item = []
        for _ in range(self.length // self.batch_size):
            random_ind = np.random.permutation(range(self.length))
            self.random_item.append(random_ind[:self.batch_size])

    def get_random_item(self, item):
        return self.random_item[item // self.batch_size][item % self.batch_size]

    def __getitem__(self, item):
        item = self.get_random_item(item)
        img = txt = None
        if self.img_read:
            img = self.read_img(item)
        if self.txt_read:
            # txt = torch.Tensor(self.txt[item][np.newaxis, :, np.newaxis])
            txt = self.read_txt(item)
        label = torch.Tensor(self.label[item])
        index = torch.from_numpy(np.array(item))
        back_dict = {'index': index, 'label': label}
        if self.img_read:
            back_dict['img'] = img
        if self.txt_read:
            back_dict['txt'] = txt
        return back_dict
        # if self.img_read is False:
        #     return {'index': index, 'txt': txt, 'label': label}
        # if self.txt_read is False:
        #     return {'index': index, 'img': img, 'label': label}
        # return {'index': index, 'img': img, 'txt': txt, 'label': label}
