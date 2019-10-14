# -*- coding: utf-8 -*-
# @Time    : 2019/7/10
# @Author  : Godder
# @Github  : https://github.com/WangGodder
import warnings
methods = ['DCMH', 'TDH', 'CMHH', 'ASCHN', 'PRDH', 'SSAH', 'RDCMH', 'ASCHN_nMS', 'ASCHN focal', 'QDCMH', 'CDQ', 'CHN', 'BCDH', 'MCDH']


__all__ = ['get_train']


@DeprecationWarning
def get_train(method: str,  dataset_name: str, img_dir: str, bit: int, **kwargs):
    warnings.warn("method 'get_train' in training package will be deprecated, please use method 'get_train' in torchcmh package")
    if method.lower() == methods[0].lower():
        from torchcmh.training.DCMH import train
    elif method.lower() == methods[1].lower():
        from torchcmh.training.TDH import train
    elif method.lower() == methods[2].lower():
        from torchcmh.training.CMHH_old import train
    elif method.lower() == methods[3].lower():
        from torchcmh.training.SCAHN import train
    elif method.lower() == methods[4].lower():
        from torchcmh.training.PRDH_old import train
    elif method.lower() == methods[5].lower():
        from torchcmh.training.SSAH import train
    elif method.lower() == methods[6].lower():
        from torchcmh.training.RDCMH import train
    elif method.lower() == methods[7].lower():
        from torchcmh.training.ASCHN_nMS import train
    elif method.lower() == methods[8].lower():
        from torchcmh.training.ASCHN_focal import train
    elif method.lower() == methods[9].lower():
        from torchcmh.training.QDCMH import train
    elif method.lower() == methods[10].lower():
        from torchcmh.training.CDQ import train
    elif method.lower() == methods[11].lower():
        from torchcmh.training.CHN import train
    elif method.lower() == methods[12].lower():
        from torchcmh.training.BCDH import train
    elif method.lower() == methods[13].lower():
        from torchcmh.training.MCDH import train
    else:
        raise ValueError("there is no method name is %s" % method)
    return train(dataset_name, img_dir, bit, **kwargs)
