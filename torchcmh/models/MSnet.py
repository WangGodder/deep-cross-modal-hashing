# -*- coding: utf-8 -*-
# @Time    : 2019/6/20
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import interpolate

from torchcmh.models import BasicModule

SEMANTIC_EMBED = 512


class MS_Block(nn.Module):
    def __init__(self, in_channel, out_channel, pool_level, txt_length):
        super(MS_Block, self).__init__()
        self.txt_length = txt_length
        pool_kernel = (5 * pool_level, 1)
        pool_stride = (5 * pool_level, 1)
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_stride)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self._init_weight()

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.relu(x)
        # resize to original size of input
        x = interpolate(x, size=(self.txt_length, 1))
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class MS_Text(BasicModule):
    def __init__(self, input_dim, output_dim):
        super(MS_Text, self).__init__()
        self.module_name = "Multi-FusionTextNet"
        self.block1 = MS_Block(1, 1, 10, input_dim)
        self.block2 = MS_Block(1, 1, 6, input_dim)
        self.block3 = MS_Block(1, 1, 3, input_dim)
        self.block4 = MS_Block(1, 1, 2, input_dim)
        self.block5 = MS_Block(1, 1, 1, input_dim)
        self.conv1 = nn.Conv2d(6, 4096, kernel_size=(input_dim, 1))
        self.LRN = nn.LocalResponseNorm(2, beta=0.75, alpha=0.0001, k=2.00)
        self.conv2 = nn.Conv2d(4096, SEMANTIC_EMBED, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(SEMANTIC_EMBED, output_dim, kernel_size=(1, 1))

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, feature_out=False):
        block1 = self.block1(x)
        block2 = self.block2(x)
        block3 = self.block3(x)
        block4 = self.block4(x)
        block5 = self.block5(x)
        ms = torch.cat([x, block1, block2, block3, block4, block5], dim=1)
        x = self.conv1(ms)
        x = F.relu(x)
        x = self.LRN(x)
        x = self.conv2(x)
        feature = F.relu(x)
        x = self.LRN(feature)
        out = self.conv3(x)
        feature = feature.squeeze()
        out = out.squeeze()
        if feature_out:
            return out, feature
        return out


def get_text_net(input_dim, output_dim):
    return MS_Text(input_dim, output_dim)


if __name__ == '__main__':
    x = torch.randn(4, 1, 1386, 1)
    net = get_text_net(1386, 24)
    y = net(x)
    print(y)
