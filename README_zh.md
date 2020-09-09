# deep cross modal hashing (torchcmh)

torchcmh是一个基于PyTorch的深度跨模态hashing库

包含:
- 数据可视化
- 17-19年有名的baseline方法
- 多个数据集读取的API
- 损失函数API
- 配置调用
-----
### 数据集

包括了4个我自己制作的数据集(Mirflickr25k, Nus Wide, MS coco, IAPR TC-12)
如果需要使用可以下载对应的.mat文件并去官网下载对应的数据集
下载方式见数据集目录下的[readme](./torchcmh/dataset/README.md)
-------
### 模型

你可以自己创建自己的模型，并且我也提供了一些与训练的模型，具体参考[README.md](./torchcmh/models/README.md)
