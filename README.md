# torch cross modal (torchcmh)

torchcmh is a library built on PyTorch for deep learning cross modal hashing.\
if you want use the dataset i make, please download mat file and image file by readme file in dataset package.
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
- change the method name in config .yml file and run.
