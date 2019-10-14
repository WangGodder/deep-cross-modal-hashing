# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder
import torch
from torch import nn
from torch.nn.functional import interpolate
from torchcmh.models import BasicModule


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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), stride=stride,
                     padding=(0, 1), bias=False)


class Release_Block(nn.Module):
    def __init__(self, kernel_size, in_channel=4096, out_channel=4096):
        super(Release_Block, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel)
        self.batch_norm1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channel)
        self.downsample = nn.Conv2d(in_channel, out_channel, kernel_size, bias=False)
        self.batch_norm_down = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)
        residual = self.downsample(residual)
        residual = self.batch_norm_down(residual)
        out += residual
        out = self.relu(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MS_Text(BasicModule):
    def __init__(self, txt_length, bit, **kwargs):
        super(MS_Text, self).__init__()
        self.module_name = 'MS Text'
        self.out_feature = kwargs['out_feature'] if 'out_feature' in kwargs.keys() else False

        # MS blocks
        self.block1 = MS_Block(1, 1, 10, txt_length)
        self.block2 = MS_Block(1, 1, 6, txt_length)
        self.block3 = MS_Block(1, 1, 3, txt_length)
        self.block4 = MS_Block(1, 1, 2, txt_length)
        self.block5 = MS_Block(1, 1, 1, txt_length)

        # full-conv layers
        self.release1 = Release_Block((txt_length, 1), 1, 64)
        self.release2 = Release_Block((1, 3), 64, 128)
        self.release3 = Release_Block((1, 3), 128, 256)
        self.release4 = Release_Block((1, 2), 256, 512)
        self.relu = nn.ReLU(inplace=True)

        self.linear_conv4 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        self.LRN = nn.LocalResponseNorm(2, alpha=0.0001, beta=0.75, k=1.)
        self.BN4 = nn.BatchNorm2d(256)
        self.hash_conv4 = nn.Conv2d(256, bit, kernel_size=(1, 1), stride=(1, 1))

        self._init_params()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.weight.weight, 0.25)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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

    def _get_ms_feature(self, x):
        """
        return all MS block out concat in dim 3
        :param x:
        :return: a tensor with shape (N, 1, text_length, 6)
        """
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        x4 = self.block4(x)
        x5 = self.block5(x)
        ms_out = torch.cat([x, x1, x2, x3, x4, x5], dim=3)
        return ms_out       # type: torch.Tensor

    def _get_feature(self, x):
        x = self.release1(x)
        x = self.release2(x)
        x = self.release3(x)
        x = self.release4(x)
        return x

    def _get_hash(self, x):
        x = self._get_feature(x)

        f = self.linear_conv4(x)
        h = self.relu(f)
        h = self.LRN(h)
        h = self.hash_conv4(h)
        # h = torch.tanh(h)
        h = h.squeeze()   # type: torch.Tensor
        if self.out_feature and self.training:
            f = f.squeeze()
            return h, f

        return h

    def forward(self, x):
        ms_out = self._get_ms_feature(x)
        return self._get_hash(ms_out)


def get_MS_Text(tag_length, bit, **kwargs):
    return MS_Text(tag_length, bit, **kwargs)


if __name__ == '__main__':
    net = MS_Text(1386, 64)
    x = torch.randn(4, 1, 1386, 1)
    y = net(x)
    print(y.shape)
    # print(y1)
