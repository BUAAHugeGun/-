import torch
from torch.utils.data import DataLoader
from torchvision import datasets as tv_datasets
from torchvision import transforms


def build_data(tag, batch_size, training, num_worker):
    if tag == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        mnist = tv_datasets.MNIST(root="../data/", train=training, transform=transform, download=True)
        dataloader = DataLoader(mnist, batch_size, shuffle=True, num_workers=num_worker)
    return dataloader
