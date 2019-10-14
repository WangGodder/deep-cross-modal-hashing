# -*- coding: utf-8 -*-
# @Time    : 2019/7/24
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import yaml
from torchvision import transforms
from torchcmh.dataset.transforms.bow import *


class Config(object):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = yaml.load(f)
            self.training = config['training']  # type: dict
            self.dataset_path = config['datasetPath']   # type: dict
            self.data_preprocess = config['dataPreprocess']
            self.data_augmentation = config['dataAugmentation']
        self.dataset_names = [key.lower() for key in self.dataset_path.keys()]
        self.dataset_path = [value for value in self.dataset_path.values()]
        self.img_training_transform = self.__get_img_train_transform()
        self.txt_training_transform = self.__get_txt_train_transform()
        self.img_valid_transform = self.__get_img_valid_transform()
        self.txt_valid_transform = self.__get_txt_valid_transform()

    def __get_img_valid_transform(self):
        transform_list = []
        data_preprocess = self.data_preprocess['img']
        resize = data_preprocess['resize']
        resize_transform = transforms.Resize(resize)
        transform_list.append(resize_transform)
        if data_preprocess['toTensor']:
            transform_list.append(transforms.ToTensor())
        mean = data_preprocess['normalize']['mean']
        std = data_preprocess['normalize']['std']
        normalize_transform = transforms.Normalize(mean, std)
        transform_list.append(normalize_transform)
        transform = transforms.Compose(transform_list)
        return transform

    # not finish
    def __get_txt_valid_transform(self):
        return None

    def __get_img_train_transform(self):
        transform_list = []
        data_preprocess = self.data_preprocess['img']
        resize = data_preprocess['resize']
        resize_transform = transforms.Resize(resize)
        transform_list.append(resize_transform)
        data_augmentation = self.data_augmentation['img']
        if self.data_augmentation["enable"]:
            if data_augmentation["randomRotation"]["enable"]:
                transforms.RandomRotation([90, 90])
        if data_preprocess['toTensor']:
            transform_list.append(transforms.ToTensor())
        mean = data_preprocess['normalize']['mean']
        std = data_preprocess['normalize']['std']
        normalize_transform = transforms.Normalize(mean, std)
        transform_list.append(normalize_transform)
        transform = transforms.Compose(transform_list)
        return transform

    def __get_txt_train_transform(self):
        transform_list = []
        data_augmentation = self.data_augmentation['txt']
        if self.data_augmentation["enable"]:
            if data_augmentation['RandomErasure']['enable']:
                prob = float(data_augmentation['RandomErasure']['probability'])
                value = float(data_augmentation['RandomErasure']['defaultValue'])
                random_erasure = RandomErasure(prob, value)
                transform_list.append(random_erasure)
        transform = transforms.Compose(transform_list)
        return transform

    def get_dataset_path(self, dataset_name: str):
        if dataset_name.lower() not in self.dataset_names:
            raise ValueError("there are not dataset name is %s" % dataset_name)
        paths = self.dataset_path[self.dataset_names.index(dataset_name.lower())]
        return paths['img_dir'], paths["imgList"], paths["tagList"], paths["labelList"]

    def get_img_dir(self):
        return self.get_dataset_path(self.training['dataName'])[0]
