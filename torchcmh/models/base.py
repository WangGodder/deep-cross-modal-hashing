# -*- coding: utf-8 -*-
# @Time    : 2019/7/22
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torch import nn
from torch.utils import model_zoo
import torch
import math
from torchcmh.models.GCN import GraphConvolution
__all__ = ['BasicModule']


class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.module_name = str(type(self))

    def init_pretrained_weights(self, model_url):
        """
        Initializes model with pretrained weights.
         Layers that don't match with pretrained layers in name or size are kept unchanged.
        :param model_url: a http url or local file path.
        :return:
        """
        if model_url[:4] == "http":
            pretrain_dict = model_zoo.load_url(model_url)
        else:
            pretrain_dict = torch.load(model_url)
        model_dict = self.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        self.load_state_dict(model_dict)
        print('Initialized model with pretrained weights from {}'.format(model_url))

    def save_entire(self, model_path):
        torch.save(self, model_path)

    def save_dict(self, model_path):
        torch.save(self.state_dict(), model_path)

    def save_state(self, model_path, epoch):
        torch.save({"state_dict": self.state_dict(), "epoch": epoch}, model_path)

    def resume_state(self, model_path):
        model_CKPT = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(model_CKPT['state_dict'])
        return model_CKPT['epoch']

    def load_dict(self, model_path):
        model_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(model_dict)

    @staticmethod
    def glorot(tensor):
        if tensor is not None:
            stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
            tensor.data.uniform_(-stdv, stdv)

    def init(self):
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
            elif isinstance(m, GraphConvolution):
                self.glorot(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, *x):
        pass

