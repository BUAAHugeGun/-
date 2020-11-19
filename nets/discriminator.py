import torch
from torch import nn


class MNIST_D(nn.Module):
    def __init__(self):
        super(MNIST_D, self).__init__()
        """
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
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 11),
        )

    def forward(self, x):
        """
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x.view(-1)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_D(tag):
    if tag == "mnist":
        return MNIST_D()


if __name__ == "__main__":
    a = torch.randn([3, 1, 28, 28])
    D = get_D("mnist")
    print(D(a))
