# -*- coding: utf-8 -*-
# @Time    : 2019/10/29
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torchcmh.models import BasicModule
from torchcmh.models.GCN import GraphConvolution
from torch import nn
import torch


class GCN(BasicModule):
    hidden_layer_dim = 1024

    def __init__(self, bit, label_dim):
        super(GCN, self).__init__()
        self.module_name = "GCH.GCN"
        self.gcn1 = GraphConvolution(bit, self.hidden_layer_dim)
        self.gcn2 = GraphConvolution(self.hidden_layer_dim, bit)
        self.hash_layer = nn.Linear(bit, bit)
        self.label_layer = nn.Linear(bit, label_dim)

    def forward(self, x, adjacent_matrix):
        feature = self.gcn1(x, adjacent_matrix)
        feature = torch.relu_(feature)
        feature = self.gcn2(feature, adjacent_matrix)
        feature = torch.sigmoid(feature)
        # hash represent
        hash_represent = self.hash_layer(feature)
        # label generation
        label = self.label_layer(hash_represent)
        # activate
        hash_represent = torch.tanh(hash_represent)
        label = torch.sigmoid(label)
        return hash_represent, label
