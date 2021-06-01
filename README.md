
# Pytorch model building tools

#### introduction
This tool can quickly build models, fine tune, output graphs, excel, models and so on.

#### environment
Pytorch 1.8.0

cuda 11.1

Python 3.6

#### How to use?

1.  get a model, you can make one or copy one, and replace classifier. such as:
```python
import torch.nn as nn
from torchvision.models import resnet34
model = resnet34(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(),
    nn.Linear(512, 87, bias=False),
)
```

2.  Load model into my tool class(ModelUtils)
```python
from utils import ModelUtils
from demo.gemstones_dataset import train_dataloader, test_dataloader
utils = ModelUtils().build(model).compose(lr=1e-3)
utils.open_checkpoint_output_image(out_dir="./out/resnet/").open_checkpoint(only_best=True)
utils.train(epochs=20, train_dataloader=train_dataloader, test_dataloader=test_dataloader)
```

3.  run this python 
``` console
training epoch[0]: 100%|██████████| 179/179 [00:20<00:00,  8.83it/s, train_all_loss=3.65, train_loss=2.28, train_rank1=0.184, train_rank5=0.392]
evaluate epoch[0]: 100%|██████████| 23/23 [00:01<00:00, 13.86it/s, test_all_loss=2.05, test_loss=2.3, test_rank1=0.512, test_rank5=0.837]
[info]正在保存模型...
[info]正在输出相关指标变化图...
training epoch[1]: 100%|██████████| 179/179 [00:17<00:00, 10.01it/s, train_all_loss=1.88, train_loss=2.91, train_rank1=0.501, train_rank5=0.846]
evaluate epoch[1]: 100%|██████████| 23/23 [00:01<00:00, 14.06it/s, test_all_loss=1.17, test_loss=1.37, test_rank1=0.631, test_rank5=0.948]
```

#### A Sample, Gemsones Classfication
Already dowmload dataset in ```dataset/Gemstones.7z```, and you need to unzip
##### MyDataset
if your dataset like this:
```
dataset_name
   ├─test
   │  ├─ class 1
   │  ├─ class 2
   │  ├─ ...
   └─train
      ├─ class 1
      ├─ class 2
      ├─ ...
```
you only need to modify **data_path** in ```demo/gemstones_dataset.py```




