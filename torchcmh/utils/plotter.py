# -*- coding: utf-8 -*-
# @Time    : 2019/5/11
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import numpy as np
from visdom import Visdom


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='plotter'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        self.epoch = 0

    def plot(self, var_name, split_name, y, x=None):
        if x is None:
            x = self.epoch
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update='append')

    def next_epoch(self):
        self.epoch += 1

    def reset_epoch(self):
        self.epoch = 0

def get_plotter(env_name: str):
    return VisdomLinePlotter(env_name)

