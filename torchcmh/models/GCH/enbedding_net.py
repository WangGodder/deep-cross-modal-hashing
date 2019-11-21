# -*- coding: utf-8 -*-
# @Time    : 2019/10/29
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torchcmh.models import BasicModule
from torch import nn
import torch
__all__ = ['get_label_net', 'get_txt_net']


class EmbeddingNet(BasicModule):
    hidden_dim1 = 512
    hidden_dim2 = 512

    def __init__(self, embedding_dim, bit, label_dim):
        super(EmbeddingNet, self).__init__()
        fc1 = nn.Linear(embedding_dim, self.hidden_dim1)
        fc2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        fc3 = nn.Linear(self.hidden_dim2, bit)
        relu = nn.ReLU(inplace=True)
        self.fc = nn.Sequential(fc1, relu, fc2, relu, fc3)
        self.label_layer = nn.Linear(bit, label_dim)

    def forward(self, x: torch.Tensor):
        if len(x.shape) > 2:
            x = x.squeeze()
        hash_represent = self.fc(x)
        if self.training:
            label_generation = self.label_layer(hash_represent)
            hash_represent = torch.tanh(hash_represent)
            label_generation = torch.sigmoid(label_generation)
            return hash_represent, label_generation
        else:
            return hash_represent


def get_label_net(label_dim, bit):
    label_net = EmbeddingNet(label_dim, bit, label_dim)
    label_net.module_name = "GCH.LabelNet"
    return label_net


def get_txt_net(txt_dim, bit, label_dim):
    txt_net = EmbeddingNet(txt_dim, bit, label_dim)
    txt_net.module_name = "GCH.TextNet"
    return txt_net
