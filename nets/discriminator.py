import torch
from torch import nn


class MNIST_D(nn.Module):
    def __init__(self):
        super(MNIST_D, self).__init__()
        layers = [
            nn.Conv2d(1, 2, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3, 1, 1),
            nn.ReLU(),
        ]
        self.conv = nn.Sequential(*layers)
        layers = [
            nn.Linear(3136, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        ]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


def get_D(tag):
    if tag == "mnist":
        return MNIST_D()


if __name__ == "__main__":
    a = torch.randn([3, 1, 28, 28])
    D = get_D("mnist")
    print(D(a))
