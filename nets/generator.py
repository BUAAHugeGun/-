import torch
from torch import nn


class MNIST_G(nn.Module):
    def __init__(self):
        super(MNIST_G, self).__init__()
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.ReLU()
        )

    def forward(self, x):
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x.clamp(0, 1)


def get_G(tag):
    if tag == "mnist":
        return MNIST_G()


if __name__ == "__main__":
    a = torch.randn([4, 1, 56, 56])
    G = get_G("mnist")
    print(G(a).shape)
