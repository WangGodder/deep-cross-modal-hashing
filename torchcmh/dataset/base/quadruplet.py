# -*- coding: utf-8 -*-
# @Time    : 2019/7/10
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from .base import CrossModalTrainBase
import numpy as np
import torch


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = label1.matmul(label2.transpose(0, 1)) > 0
    return Sim.float()


class CrossModalQuadrupletTrain(CrossModalTrainBase):
    """
    Quadruplet: return one sample(s), one positive(p), two negative(n1, n2) which n1 and n2 are negative each other.
    """
    def __init__(self, img_dir: str, img_names: np.ndarray, txt_matrix: np.ndarray, label_matrix: np.ndarray,
                 img_transform, batch_size):
        super(CrossModalQuadrupletTrain, self).__init__(img_dir, img_names, txt_matrix, label_matrix, img_transform)
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

    def __get_positive_index(self, ind, include_self=False):
        current_sim = self.sim[ind]
        index = torch.nonzero(current_sim)
        if len(index.shape) > 1:
            index = index.reshape(-1)
        if include_self:
            return index.numpy()
        index = np.setdiff1d(index.numpy(), ind)
        return index

    def re_random_item(self):
        self.random_item = []
        for _ in range(self.length // self.batch_size):
            random_ind = np.random.permutation(range(self.length))
            self.random_item.append(random_ind[:self.batch_size])

    def get_random_item(self, item):
        return self.random_item[item // self.batch_size][item % self.batch_size]

    def _get_random_quadruplet_index(self, query_ind):
        """
        randomly get a positive instance and two negative instances which two negative instances are not similar from train set
        :param query_ind: the index of query instance in train indexes
        :return: positive index and two negative indexes in train indexes
        """
        pos_indexes = self.triplet_indexes[query_ind][0]
        neg_indexes = self.triplet_indexes[query_ind][1]
        pos_ind = np.random.choice(pos_indexes)
        neg_ind1 = np.random.choice(neg_indexes)
        neg_ind2 = np.random.choice(np.setdiff1d(neg_indexes, self.__get_positive_index(neg_ind1, True)))
        return pos_ind, neg_ind1, neg_ind2

    def __getitem__(self, item):
        """
        item dataset return query instance with M1 positive instances and M2 negative instances
        if use DataLoader to get item, then return of positive(negative) with shape (batch size, M1(2), model shape)
        :param item:
        :return:
        """
        query_ind = self.get_random_item(item)
        positive_ind, negative_ind1, negative_ind2 = self._get_random_quadruplet_index(query_ind)
        if self.img_read:
            img = self.read_img(query_ind)
            pos_img = self.read_img(positive_ind)
            neg_img1 = self.read_img(negative_ind1)
            neg_img2 = self.read_img(negative_ind2)
        if self.txt_read:
            txt = torch.Tensor(self.txt[query_ind][np.newaxis, :, np.newaxis])
            pos_txt = torch.Tensor(self.txt[positive_ind][np.newaxis, :, np.newaxis])
            neg_txt1 = torch.Tensor(self.txt[negative_ind1][np.newaxis, :, np.newaxis])
            neg_txt2 = torch.Tensor(self.txt[negative_ind2][np.newaxis, :, np.newaxis])
        label = torch.Tensor(self.label[query_ind])
        query_ind = torch.from_numpy(np.array(query_ind))
        positive_ind = torch.from_numpy(np.array(positive_ind))
        negative_ind1 = torch.from_numpy(np.array(negative_ind1))
        negative_ind2 = torch.from_numpy(np.array(negative_ind2))
        if self.img_read is False:
            return {'index': query_ind, 'pos_index': positive_ind, 'neg_index1': negative_ind1,
                    'neg_index2': negative_ind2,
                    'txt': txt, 'pos_txt': pos_txt, 'neg_txt1': neg_txt1, 'neg_txt2': neg_txt2, 'label': label}
        if self.txt_read is False:
            return {'index': query_ind, 'pos_index': positive_ind, 'neg_index1': negative_ind1,
                    'neg_index2': negative_ind2,
                    'img': img, 'pos_img': pos_img, 'neg_img1': neg_img1, 'neg_img2': neg_img2, 'label': label}
        return {'index': query_ind, 'pos_index': positive_ind, 'neg_index1': negative_ind1, 'neg_index2': negative_ind2,
                'txt': txt, 'pos_txt': pos_txt, 'neg_txt1': neg_txt1, 'neg_txt2': neg_txt2,
                'img': img, 'pos_img': pos_img, 'neg_img1': neg_img1, 'neg_img2': neg_img2, 'label': label}
