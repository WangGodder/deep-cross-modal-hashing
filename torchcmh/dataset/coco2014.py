# -*- coding: utf-8 -*-
# @Time    : 2019/6/27
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from .base import *
from scipy import io as sio
import numpy as np
from torchcmh.dataset import abs_dir
import os


default_img_mat_url = os.path.join(abs_dir, "data", "coco2014", "imgList.mat")
default_tag_mat_url = os.path.join(abs_dir, "data", "coco2014", "tagList.mat")
default_label_mat_url = os.path.join(abs_dir, "data", "coco2014", "labelList.mat")
default_seed = 8

img_names = None
txt = None
label = None


def load_mat(img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url):
    img_names = sio.loadmat(img_mat_url)['imgs']  # type: np.ndarray
    img_names = img_names.squeeze()
    all_img_names = img_names
    all_txt = np.array(sio.loadmat(tag_mat_url)['tags'], dtype=np.float)
    all_label = np.array(sio.loadmat(label_mat_url)['labels'], dtype=np.float)
    return all_img_names, all_txt, all_label


def split_data(all_img_names, all_txt, all_label, query_num=5000, train_num=10000, seed=None):
    np.random.seed(seed)
    random_index = np.random.permutation(range(all_img_names.shape[0]))
    query_index = random_index[: query_num]
    train_index = random_index[query_num: query_num + train_num]
    retrieval_index = random_index[query_num:]

    query_img_names = all_img_names[query_index]
    train_img_names = all_img_names[train_index]
    retrieval_img_names = all_img_names[retrieval_index]

    query_txt = all_txt[query_index]
    train_txt = all_txt[train_index]
    retrieval_txt = all_txt[retrieval_index]

    query_label = all_label[query_index]
    train_label = all_label[train_index]
    retrieval_label = all_label[retrieval_index]

    img_names = (query_img_names, train_img_names, retrieval_img_names)
    txt = (query_txt, train_txt, retrieval_txt)
    label = (query_label, train_label, retrieval_label)
    return img_names, txt, label


def load_dataset(train_dataset, img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url,
                 label_mat_url=default_label_mat_url, batch_size=128, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
    global img_names, txt, label
    if img_names is None:
        all_img_names, all_txt, all_label = load_mat(img_mat_url, tag_mat_url, label_mat_url)
        img_names, txt, label = split_data(all_img_names, all_txt, all_label, query_num, train_num, seed)
        print("COCO2014 data load and shuffle by seed %d" % seed)
    img_train_transform = kwargs['img_train_transform'] if 'img_train_transform' in kwargs.keys() else None
    txt_train_transform = kwargs['txt_train_transform'] if 'txt_train_transform' in kwargs.keys() else None
    img_valid_transform = kwargs['img_valid_transform'] if 'img_valid_transform' in kwargs.keys() else None
    txt_valid_transform = kwargs['txt_valid_transform'] if 'txt_valid_transform' in kwargs.keys() else None
    train_data = train_dataset(img_dir, img_names[1], txt[1], label[1], batch_size, img_train_transform, txt_train_transform)
    valid_data = CrossModalValidBase(img_dir, img_names[0], img_names[2], txt[0], txt[2], label[0], label[2], img_valid_transform,
                                     txt_valid_transform)
    return train_data, valid_data


def get_single_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
                        batch_size=128, train_num=10000, query_num=5000, seed=default_seed, **kwargs):
    print("load data set single MS COCO")
    return load_dataset(CrossModalSingleTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size, train_num=train_num,
                        query_num=query_num, seed=seed, **kwargs)


def get_pairwise_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
                          batch_size=128, train_num=10000, query_num=5000, seed=default_seed, **kwargs):
    print("load data set pairwise MS COCO")
    return load_dataset(CrossModalPairwiseTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
                        train_num=train_num, query_num=query_num, seed=seed, **kwargs)


def get_triplet_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
                         batch_size=128, train_num=10000, query_num=5000, seed=default_seed, **kwargs):
    print("load data set triplet MS COCO")
    return load_dataset(CrossModalTripletTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
                        train_num=train_num, query_num=query_num, seed=seed, **kwargs)


def get_quadruplet_datasets(img_dir, img_mat_url=default_img_mat_url, tag_mat_url=default_tag_mat_url, label_mat_url=default_label_mat_url,
                            batch_size=128, train_num=10000, query_num=2000, seed=default_seed, **kwargs):
    print("load data set quadruplet MS COCO")
    return load_dataset(CrossModalQuadrupletTrain, img_dir, img_mat_url, tag_mat_url, label_mat_url, batch_size=batch_size,
                        train_num=train_num, query_num=query_num, seed=seed, **kwargs)


__all__ = ['get_single_datasets', 'get_pairwise_datasets', 'get_triplet_datasets', 'get_quadruplet_datasets']
