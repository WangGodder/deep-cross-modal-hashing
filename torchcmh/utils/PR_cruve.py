# -*- coding: utf-8 -*-
# @Time    : 2019/8/4
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import scipy.io as sio
from torchcmh.training.valid import precision_recall, PR_curve
from torchcmh.dataset import single_data
import os


def load_hash(path):
    hash_mat = []
    labels = []
    for file in os.listdir(path):
        hash_mat.append(sio.loadmat(os.path.join(path, file)))
        labels.append(file.split(".")[0].split('-')[1])
    return hash_mat, labels


def calc_value(hash_mat, data_name):
    _, valid_data = single_data(data_name, "./")
    valid_data.query()
    query_label = valid_data.get_all_label()
    valid_data.retrieval()
    retrieval_label = valid_data.get_all_label()
    i2ts = t2is = []
    for mat in hash_mat:
        query_img = mat['q_img']
        query_txt = mat['q_txt']
        retrieval_img = mat['r_img']
        retrieval_txt = mat['r_txt']
        i2t = precision_recall(query_img, retrieval_txt, query_label, retrieval_label, len(valid_data))
        t2i = precision_recall(query_txt, retrieval_img, query_label, retrieval_label, len(valid_data))
        i2ts.append(i2t)
        t2is.append(t2i)
    return i2ts, t2is


def draw_PR(path, data_name):
    hash_mat, labels = load_hash(path)
    i2ts, t2is = calc_value(hash_mat, data_name)
    PR_curve(i2ts, labels)
    PR_curve(t2is, labels)






