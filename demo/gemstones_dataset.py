import torch.nn as nn
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.models import resnet34, resnet18
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
from utils import ModelUtils
import warnings
warnings.filterwarnings("ignore")


# training's transform
transform = transforms.Compose(
    [

        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=[-30, 30]),
        transforms.ToTensor(),
    ])

# test's transform
test_transform = transforms.Compose(
    [
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


class ImageDataset(Dataset):

    def __init__(self, root, is_test=False):
        self.dataset = ImageFolder(root=root)
        imgs = np.array(self.dataset.imgs)
        self.imgPath = imgs[:, 0]
        self.label = imgs[:, 1].astype(np.int64)
        self.label = torch.from_numpy(self.label)
        self.is_test = is_test

    def __getitem__(self, index):
        img = Image.open(self.imgPath[index])
        if self.is_test:
            img = test_transform(img)
        else:
            img = transform(img)
        label = self.label[index]
        # if img.shape[0] > 3:
        #     print(self.imgPath[index])
        return img, label

    def __len__(self):
        return len(self.imgPath)

# Please remember unzip Gemstones.7z
data_path = r"../dataset/Gemstones"
batch_size = 16
test_batch_size = 16


train_dataset = ImageDataset(root=data_path + r"/train")
test_dataset = ImageDataset(root=data_path + r"/test", is_test=True)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True, pin_memory=True)


if __name__ == '__main__':
    pass
