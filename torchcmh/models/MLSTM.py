# -*- coding: utf-8 -*-
# @Time    : 2019/12/28
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torch import nn
from torchcmh.models import BasicModule
import torch


class MultiLSTM(BasicModule):
    def __init__(self, input_dim, output_dim, hidden_dims, bi_lstm=False, bn=True, leaky_relu=None, dropout=0):
        super(MultiLSTM, self).__init__()
        self.module_name = "Multi-LSTM"
        self.layers = []
        seq_length = input_dim
        for hidden_dim in hidden_dims:
            if bi_lstm:
                hidden_dim //= 2
            self.layers.append(nn.LSTM(seq_length, hidden_dim, 1, dropout=dropout, bidirectional=bi_lstm))
            seq_length = hidden_dim
            if leaky_relu:
                self.layers.append(nn.LeakyReLU(leaky_relu, inplace=True))
            else:
                self.layers.append(nn.ReLU(inplace=True))
        self.fc = nn.Conv1d(seq_length, output_dim, kernel_size=1, stride=1)

    def forward(self, x, out_feature=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if len(x.shape) > 3:
            x = x.squeeze().unsqueeze(1)
        for layer in self.layers:
            if isinstance(layer, nn.RNNBase):
                x, _ = layer(x)
            else:
                x = layer(x)
        x = x.transpose(1, 2)
        out = self.fc(x)
        out = out.squeeze()
        if out_feature:
            return out, x.squeeze()
        return out


if __name__ == '__main__':
    x = torch.randn(4, 24)
    net = MultiLSTM(24, 4, [64, 256, 64])
    y = net(x)
    print(y.shape)