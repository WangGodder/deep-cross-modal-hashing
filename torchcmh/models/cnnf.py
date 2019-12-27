# -*- coding: utf-8 -*-
# @Time    : 2019/7/22
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import torch.nn as nn
from torchcmh.models import abs_dir, BasicModule
import os

__all__ = ['get_cnnf', 'CNNF']

pretrain_model = os.path.join(abs_dir, "pretrain_model", "imagenet-vgg-f.pth")


class CNNF(BasicModule):
    def __init__(self, bit, leakRelu=False, bn=False):
        super(CNNF, self).__init__()
        self.module_name = "CNNF"
        self.features = nn.Sequential(
            # 0 conv1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4),
            # 1 relu1
            nn.LeakyReLU(inplace=True) if leakRelu else nn.ReLU(inplace=True),
            # 2 norm1
            nn.BatchNorm2d(64) if bn else nn.LocalResponseNorm(size=2, k=2),
            # 3 pool1
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 4 conv2
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=5, stride=1, padding=2),
            # 5 relu2
            nn.LeakyReLU(inplace=True) if leakRelu else nn.ReLU(inplace=True),
            # 6 norm2
            nn.BatchNorm2d(256) if bn else nn.LocalResponseNorm(size=2, k=2),
            # 7 pool2
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 8 conv3
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 9 relu3
            nn.LeakyReLU(inplace=True) if leakRelu else nn.ReLU(inplace=True),
            # 10 conv4
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 11 relu4
            nn.LeakyReLU(inplace=True) if leakRelu else nn.ReLU(inplace=True),
            # 12 conv5
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            # 13 relu5
            nn.LeakyReLU(inplace=True) if leakRelu else nn.ReLU(inplace=True),
            # 14 pool5
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.fc = nn.Sequential(
            # 15 full_conv6
            nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=6),
            # 16 relu6
            nn.LeakyReLU(inplace=True) if leakRelu else nn.ReLU(inplace=True),
            # 17 full_conv7
            nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1),
            # 18 relu7
            nn.LeakyReLU(inplace=True) if leakRelu else nn.ReLU(inplace=True),
        )
        # fc8
        self.classifier = nn.Linear(in_features=4096, out_features=bit)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, out_feature=False):
        x = self.features(x)
        out = self.fc(x)
        feature = out.squeeze()
        out = self.classifier(feature)
        if out_feature:
            return out, feature
        return out


def get_cnnf(bit, pretrain=True, leakRelu=False, bn=False):
    model = CNNF(bit, leakRelu=leakRelu, bn=bn)
    if pretrain:
        model.init_pretrained_weights(pretrain_model)
    return model

