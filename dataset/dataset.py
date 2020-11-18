import torch
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms


def build_data(tag, batch_size):
    if tag == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        mnist = tv_datasets.MNIST(root="../data/", train=True, transform=transform)
        dataloader = DataLoader(mnist, batch_size, shuffle=True)
    return dataloader
