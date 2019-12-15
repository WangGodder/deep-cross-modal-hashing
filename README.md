# torch cross modal (torchcmh)

torchcmh is a library built on PyTorch for deep learning cross modal hashing.\
if you want use the dataset i make, please download mat file and image file by readme file in dataset package.

you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+
### how to using
- create a configuration file as ./script/default_config.yml
```yaml
training:
  method: DCMH
  dataName: Mirflickr25K
  batchSize: 64
  bit: 64
  cuda: True
  device: 0
datasetPath:
  Mirflickr25k:
    img_dir: I:\dataset\mirflickr25k\mirflickr

```
- run ./script/main.py and input configuration file path.
```python
from torchcmh.run import run
if __name__ == '__main__':
    run(config_path='default_config.yml')
```
### how to create your method
- create new method file in folder ./torchcmh/training/
- inherit implement TrainBase
- change the training.method as your python file name in config .yml file and run.

### some function in TrainBase
- data visualization
```python
def plot_loss(self, title: str, loss_store=None):
    if loss_store is None:
        loss_store = self.loss_store
    if self.plotter:
        for name, loss in loss_store.items():
            self.plotter.plot(title, name, loss.avg)
```
- valid
```python
for epoch in range(self.max_epoch):
    # training codes
    self.valid(epoch)
```
