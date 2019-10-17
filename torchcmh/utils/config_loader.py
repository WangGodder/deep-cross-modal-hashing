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
        if self.data_augmentation["enable"] and data_augmentation["enable"]:
            original_retention = float(data_augmentation['originalRetention'])
            data_augmentation_transform_list = []
            if data_augmentation["randomRotation"]["enable"]:
                rotation_list = data_augmentation["randomRotation"]["rotationAngle"]
                rotation_transforms = []
                for rotation in rotation_list:
                    rotation_transforms.append(transforms.RandomRotation(rotation))
                probability = float(data_augmentation["randomRotation"]["probability"])
                random_rotation = transforms.RandomApply(rotation_transforms, probability)
                data_augmentation_transform_list.append(random_rotation)
            if data_augmentation["RandomHorizontalFlip"]["enable"]:
                probability = data_augmentation["RandomHorizontalFlip"]["probability"]
                horizontal_flip = transforms.RandomHorizontalFlip(probability)
                data_augmentation_transform_list.append(horizontal_flip)
            if data_augmentation["RandomVerticalFlip"]["enable"]:
                probability = data_augmentation["RandomVerticalFlip"]["probability"]
                vertical_flip = transforms.RandomVerticalFlip(probability)
                data_augmentation_transform_list.append(vertical_flip)
            data_augmentation_transform = transforms.RandomApply(data_augmentation_transform_list, p=1-original_retention)
            transform_list.append(data_augmentation_transform)
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
