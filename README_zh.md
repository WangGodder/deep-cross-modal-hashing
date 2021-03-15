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

---
### 所需依赖 
你需要安装以下依赖文件，为了保证兼容性请尽量按照如下所示版本号安装
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+
----
### 日志和记录

所偶的算法模型在训练时都会自动生成日志和模型记录. \
日志在 ./logs/\{method_name\}/\{dataset_name\}/date.txt \
模型在 ./checkpoints/\{method_name\}/\{dataset_name\}/\{bit\}-\{model_name\}.pth

----
### 如何使用
- 创建一个配置文件例如  ./script/default_config.yml
```yaml
training:
  # the name of python file in training
  method: SCAHN
  # the data set name, you can choose mirflickr25k, nus wide, ms coco, iapr tc-12
  dataName: Mirflickr25K
  batchSize: 64
  # the bit of hash codes
  bit: 64
  # if true, the program will be run on gpu. Of course, you need install 'cuda' and 'cudnn' better.
  cuda: True
  # the device id you want to use, if you want to multi gpu, you can use [id1, id2]
  device: 0
datasetPath:
  Mirflickr25k:
    # the path you download the image of data set. Attention: image files, not mat file.
    img_dir: \dataset\mirflickr25k\mirflickr

```
- 将配置的路径输入并执行 ./script/main.py.
```python
from torchcmh.run import run
if __name__ == '__main__':
    run(config_path='default_config.yml')
```
- 数据可视化
在训练之前需要先打开visdom的服务，命令如下
```shell script
python -m visdom.server
```
Then you can see the charts in browser in special port.

----
### 创建自己的算法
- 在目录 ./torchcmh/training/ 下创建自己的算法
- 算法类继承TrainBase基类
- change config.yml file and run.

#### TrainBase提供的一些函数
- loss variable \
In your method, some variables you need to store and check, such as loss and acc. 
You can use var in TrainBase "loss_store" to store:
```pythn
self.loss_store = ["log loss", 'quantization loss', 'balance loss', 'loss']
```
"loss_store" is a list, push the name and update value by "loss_store\[name\].update()":
```python
value = 1000    # the value to update 
n = 10          # the number of instance for current value
self.loss_store['log loss'].update(value, n)
```
For print and visualization the loss, you can use:
```python
epoch = 1       # current epoch
self.print_loss(epoch)  # print loss
self.plot_loss("img loss")  # visualization img loss is the name of chart
```
clean "loss_store"
```python
self.reset_loss()   # reset loss_store
```
- parameters
In your method, all parameters can be stored as follows:
```python
self.parameters = {'gamma': 1, 'eta': 1}    # {name: value}
```
when method training, log will record the parameters and learning rate.
- valid
```python
for epoch in range(self.max_epoch):
    # training codes
    self.valid(epoch)
```
----
### LICENSE
this repository keep MIT license.