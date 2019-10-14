# -*- coding: utf-8 -*-
# @Time    : 2019/7/13
# @Author  : Godder
# @Github  : https://github.com/WangGodder
from torchcmh.run import run
import torch

torch.backends.cudnn.benchmark = True
mirflickr_dir = "I:\dataset\mirflickr25k\mirflickr"


# def run(method: str, dataset_name: str, img_dir: str, bit: int, **kwargs):
#     t = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
#     sys.stdout = Logger(os.path.join('..', 'logs', method.upper(), dataset_name.upper(), t + '.txt'))
#     get_train(method.upper(), dataset_name.upper(), img_dir, bit, **kwargs)


if __name__ == '__main__':
    run()
    # import scipy.io as sio
    # import numpy as np
    # from torchcmh.training.valid import precision_recall, PR_curve
    # from torchcmh.dataset import single_data
    # _, valid_data = single_data("mirflickr25k", mirflickr_dir, batch_size=32)
    # hash_code = sio.loadmat("../checkpoints/CDQ/MIRFLICKR25K/hash-16.mat")
    # query_img = torch.tensor(hash_code['q_img'])
    # query_txt = torch.tensor(hash_code['q_txt'])
    # retrieval_img = torch.tensor(hash_code['r_img'])
    # retrieval_txt = torch.tensor(hash_code['r_txt'])
    #
    # valid_data.query()
    # query_label = valid_data.get_all_label()
    # valid_data.retrieval()
    # retrieval_label = valid_data.get_all_label()
    # precisions = precision_recall(query_img, retrieval_txt, query_label, retrieval_label, len(valid_data))
    # print(precisions)
    # PR_curve(np.array([precisions]), ['CDQ'])

    # from torchcmh.dataset.utils.packing import packing_image
    # packing_image("nus wide", "I:\dataset\\NUSWIDE\Flickr", (224, 224))
    # config = Config("default_config.yml")

    # run('MCDH', 'mirflickr25k', mirflickr_dir, 32, batch_size=64)

