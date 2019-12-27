# -*- coding: utf-8 -*-
# @Time    : 2019/7/22
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torch import nn
from torchcmh.models import BasicModule
import torch


__all__ = ['MLP']


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)
    elif type(m) == nn.Conv1d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class MLP(BasicModule):
    def __init__(self, input_dim, output_dim, hidden_nodes=[8192], dropout=None, leakRelu=False, bn=False):
        """
        :param input_dim: dimension of input
        :param output_dim: bit number of the final binary code
        """
        super(MLP, self).__init__()
        self.module_name = "MLP"

        # full-conv layers
        full_conv_layers = []
        in_channel = 1
        for hidden_node in hidden_nodes:
            kernel_size = input_dim if in_channel == 1 else 1
            full_conv_layers.append(nn.Conv1d(in_channel, hidden_node, kernel_size=kernel_size, stride=1))
            in_channel = hidden_node
            if bn:
                full_conv_layers.append(nn.BatchNorm1d(hidden_node))
            if dropout:
                full_conv_layers.append(nn.Dropout(dropout))
            if leakRelu:
                full_conv_layers.append(nn.LeakyReLU(inplace=True))
            else:
                full_conv_layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*full_conv_layers)
        self.fc = nn.Conv1d(in_channel, output_dim, kernel_size=1, stride=1)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor, out_feature=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if len(x.shape) > 3:
            x = x.squeeze().unsqueeze(1)
        x = self.layers(x)
        out = self.fc(x)
        out = out.squeeze()
        if out_feature:
            return out, x.squeeze()
        return out
