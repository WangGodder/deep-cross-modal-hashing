# -*- coding: utf-8 -*-
# @Time    : 2019/6/29
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from .base import CrossModalTrainBase
import numpy as np
import torch


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = label1.matmul(label2.transpose(0, 1)) > 0
    return Sim.float()


class CrossModalTripletTrain(CrossModalTrainBase):
    def __init__(self, img_dir: str, img_names: np.ndarray, txt_matrix: np.ndarray, label_matrix: np.ndarray, batch_size,
                 img_transform, txt_transform):
        super(CrossModalTripletTrain, self).__init__(img_dir, img_names, txt_matrix, label_matrix, img_transform, txt_transform)
        self.batch_size = batch_size
        self.sim = calc_neighbor(self.get_all_label(), self.get_all_label())
        self.triplet_indexes = self.__get_triplet_indexes()
        self.re_random_item()

    def __get_triplet_indexes(self):
        indexes = []
        for ind in range(self.length):
            pos_ind = self.__get_positive_index(ind)
            neg_ind = np.setdiff1d(np.arange(self.length), pos_ind)
            neg_ind = np.setdiff1d(neg_ind, ind)
            index = [pos_ind, neg_ind]
            indexes.append(index)
        return indexes

    def __get_positive_index(self, ind):
        current_sim = self.sim[ind]
        index = torch.nonzero(current_sim)
        if len(index.shape) > 1:
            index = index.reshape(-1)
        index = np.setdiff1d(index.numpy(), ind)
        return index

    def re_random_item(self):
        self.random_item = []
        for _ in range(self.length // self.batch_size):
            random_ind = np.random.permutation(range(self.length))
            self.random_item.append(random_ind[:self.batch_size])

    def get_random_item(self, item):
        return self.random_item[item // self.batch_size][item % self.batch_size]

    def _get_random_triplet_index(self, query_ind):
        """
        randomly get a positive instance and a negative instance from train set for query instance
        :param query_ind: the index of query instance in train indexes
        :return: positive index and negative index in train indexes
        """
        pos_indexes = self.triplet_indexes[query_ind][0]
        neg_indexes = self.triplet_indexes[query_ind][1]
        pos_ind = np.random.choice(pos_indexes)
        neg_ind = np.random.choice(neg_indexes)
        return pos_ind, neg_ind

    def __getitem__(self, item):
        """
        item dataset return query instance with M1 positive instances and M2 negative instances
        if use DataLoader to get item, then return of positive(negative) with shape (batch size, M1(2), model shape)
        :param item: index of data read
        :return:
        """
        query_ind = self.get_random_item(item)
        positive_ind, negative_ind = self._get_random_triplet_index(query_ind)
        if self.img_read:
            img = self.read_img(query_ind)
            pos_img = self.read_img(positive_ind)
            neg_img = self.read_img(negative_ind)
        if self.txt_read:
            txt = self.read_txt(query_ind)
            pos_txt = self.read_txt(positive_ind)
            neg_txt = self.read_txt(negative_ind)
        label = torch.Tensor(self.label[query_ind])
        label_pos = torch.Tensor(self.label[positive_ind])
        label_neg = torch.Tensor(self.label[negative_ind])
        query_ind = torch.from_numpy(np.array(query_ind))
        positive_ind = torch.from_numpy(np.array(positive_ind))
        negative_ind = torch.from_numpy(np.array(negative_ind))
        back_dict = {'index': query_ind, 'pos_index': positive_ind, 'neg_index': negative_ind,
                     'label': label, 'pos_label': label_pos, 'neg_label': label_neg}
        if self.txt_read:
            back_dict['txt'] = txt
            back_dict['pos_txt'] = pos_txt
            back_dict['neg_txt'] = neg_txt
        if self.img_read:
            back_dict['img'] = img
            back_dict['pos_img'] = pos_img
            back_dict['neg_img'] = neg_img
        return back_dict
