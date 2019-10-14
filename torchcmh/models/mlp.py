# -*- coding: utf-8 -*-
# @Time    : 2019/7/22
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torch import nn
from torch.nn import functional as F
from torchcmh.models import BasicModule


__all__ = ['MLP']


def weights_init(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight.data, 0.0, 0.01)
        nn.init.normal_(m.bias.data, 0.0, 0.01)


class MLP(BasicModule):
    def __init__(self, y_dim, bit, out_feature=False, hidden_node=8192, dropout=None):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(MLP, self).__init__()
        self.module_name = "MLP"
        self.out_feature = out_feature

        # full-conv layers
        self.conv1 = nn.Conv2d(1, hidden_node, kernel_size=(y_dim, 1), stride=(1, 1))
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.conv2 = nn.Conv2d(hidden_node, bit, kernel_size=1, stride=(1, 1))
        self.apply(weights_init)

    def forward(self, x):
        x = self.conv1(x)
        f = F.relu(x)
        if self.dropout:
            x = self.dropout(f)
        x = self.conv2(x)
        x = x.squeeze()
        if self.out_feature:
            return x, f
        return x
