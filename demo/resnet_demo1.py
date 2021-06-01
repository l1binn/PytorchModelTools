import torch.nn as nn
from torchvision.models import resnet34
import torch
import warnings
warnings.filterwarnings("ignore")
from demo.gemstones_dataset import train_dataloader, test_dataloader
from utils import ModelUtils

if __name__ == '__main__':
    model = resnet34(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(),
        nn.Linear(512, 87, bias=False),
    )

    utils = ModelUtils().build(model).compose(lr=1e-3)
    utils.open_checkpoint_output_image(out_dir="./out/resnet/").open_checkpoint(only_best=True)
    utils.train(epochs=20, train_dataloader=train_dataloader, test_dataloader=test_dataloader)


