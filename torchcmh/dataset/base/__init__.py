# coding: utf-8
# @Time     : 
# @Author   : Godder
# @Github   : https://github.com/WangGodder
from __future__ import absolute_import
from __future__ import print_function

from .base import CrossModalTrainBase, CrossModalValidBase
from .pairwise import CrossModalPairwiseTrain
from .single import CrossModalSingleTrain
from .triplet import CrossModalTripletTrain
from .quadruplet import CrossModalQuadrupletTrain
from .triplet_ranking import CrossModalTripletRankingTrain


__all__ = ['CrossModalTrainBase', 'CrossModalValidBase', 'CrossModalSingleTrain',
           'CrossModalPairwiseTrain', 'CrossModalTripletTrain', 'CrossModalQuadrupletTrain']

