# -*- coding: utf-8 -*-
# @Time    : 2019/7/13
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import os
abs_dir = os.path.dirname(__file__)
__all__ = ['single_data', 'pairwise_data', 'triplet_data', 'triplet_rank_data', 'quadruplet_data', "abs_dir"]


def triplet_data(dataset_name: str, img_dir: str, **kwargs):
    if dataset_name.lower() == 'mirflickr25k':
        from torchcmh.dataset.mirflckr25k import get_triplet_datasets
    elif dataset_name.lower() in ['nus wide', 'nuswide']:
        from torchcmh.dataset.nus_wide import get_triplet_datasets
    elif dataset_name.lower() in ['coco2014', 'coco', 'mscoco', 'ms coco']:
        from torchcmh.dataset.coco2014 import get_triplet_datasets
    elif dataset_name.lower() in ['iapr tc-12', 'iapr', 'tc-12', 'tc12']:
        from torchcmh.dataset.tc12 import get_triplet_datasets
    else:
        raise ValueError("there is no dataset name is %s" % dataset_name)

    return get_triplet_datasets(img_dir, **kwargs)


def single_data(dataset_name: str, img_dir: str, **kwargs):
    if dataset_name.lower() == 'mirflickr25k':
        from torchcmh.dataset.mirflckr25k import get_single_datasets
    elif dataset_name.lower() in ['nus wide', 'nuswide']:
        from torchcmh.dataset.nus_wide import get_single_datasets
    elif dataset_name.lower() in ['coco2014', 'coco', 'mscoco', 'ms coco']:
        from torchcmh.dataset.coco2014 import get_single_datasets
    elif dataset_name.lower() in ['iapr tc-12', 'iapr', 'tc-12', 'tc12']:
        from torchcmh.dataset.tc12 import get_single_datasets
    else:
        raise ValueError("there is no dataset name is %s" % dataset_name)

    return get_single_datasets(img_dir, **kwargs)


def pairwise_data(dataset_name: str, img_dir: str, **kwargs):
    if dataset_name.lower() == 'mirflickr25k':
        from torchcmh.dataset.mirflckr25k import get_pairwise_datasets
    elif dataset_name.lower() in ['nus wide', 'nuswide']:
        from torchcmh.dataset.nus_wide import get_pairwise_datasets
    elif dataset_name.lower() in ['coco2014', 'coco', 'mscoco', 'ms coco']:
        from torchcmh.dataset.coco2014 import get_pairwise_datasets
    elif dataset_name.lower() in ['iapr tc-12', 'iapr', 'tc-12', 'tc12']:
        from torchcmh.dataset.tc12 import get_pairwise_datasets
    else:
        raise ValueError("there is no dataset name is %s" % dataset_name)

    return get_pairwise_datasets(img_dir, **kwargs)


def triplet_rank_data(dataset_name: str, img_dir:str, **kwargs):
    if dataset_name.lower() == 'mirflickr25k':
        from torchcmh.dataset.mirflckr25k import get_triplet_ranking_datasets
    elif dataset_name.lower() == 'nus wide':
        from torchcmh.dataset.nus_wide import get_triplet_ranking_datasets
    elif dataset_name.lower() == 'coco2014':
        from torchcmh.dataset.coco2014 import get_triplet_ranking_datasets
    elif dataset_name.lower() == 'iapr tc-12':
        from torchcmh.dataset.tc12 import get_triplet_ranking_datasets
    else:
        raise ValueError("there is no dataset name is %s" % dataset_name)

    return get_triplet_ranking_datasets(img_dir, **kwargs)


def quadruplet_data(dataset_name: str, img_dir: str, **kwargs):
    if dataset_name.lower() == 'mirflickr25k':
        from torchcmh.dataset.mirflckr25k import get_quadruplet_datasets
    elif dataset_name.lower() == 'nus wide':
        from torchcmh.dataset.nus_wide import get_quadruplet_datasets
    elif dataset_name.lower() == 'coco2014':
        from torchcmh.dataset.coco2014 import get_quadruplet_datasets
    elif dataset_name.lower() == 'iapr tc-12':
        from torchcmh.dataset.tc12 import get_quadruplet_datasets
    else:
        raise ValueError("there is no dataset name is %s" % dataset_name)

    return get_quadruplet_datasets(img_dir, **kwargs)
