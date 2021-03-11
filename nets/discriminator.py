import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models import resnet18
from torchvision.models import resnet101


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
            nn.Conv2d(3, 32, 5, padding=2),  # batch, 32, 28, 28
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2),  # batch, 32, 14, 14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 5, padding=2),  # batch, 64, 14, 14
            nn.LeakyReLU(0.2, True),
            nn.AvgPool2d(2, stride=2)  # batch, 64, 7, 7
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 1000),
            nn.LeakyReLU(0.2, True)
        )
        self.validity_layer = nn.Sequential(nn.Linear(1000, 1), nn.Sigmoid())
        self.label_layer = nn.Sequential(nn.Linear(1000, 11), nn.LogSoftmax(dim=1))

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
        validity = self.validity_layer(x).view(-1)
        plabel = self.label_layer(x).view(-1, 11)
        return validity, plabel


class Resnet(nn.Module):
    def __init__(self, classes):
        super(Resnet, self).__init__()
        self.model = resnet18()
        self.classes = classes
        self.validity_layer = nn.Sequential(nn.Linear(1000, 1), nn.Sigmoid())
        self.label_layer = nn.Sequential(nn.Linear(1000, self.classes), nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.model(x)
        validity = self.validity_layer(x).view(-1)
        plabel = self.label_layer(x).view(-1, self.classes)
        return validity, plabel


def get_D(tag, **kwargs):
    if tag == "mnist":
        return MNIST_D()
    if tag == "resnet":
        classes = kwargs.get("classes", None)
        if classes is None:
            print("resnet need parameter: classes")
            assert 0
        return Resnet(classes)


if __name__ == "__main__":
    a = torch.randn([3, 1, 64, 64])
    D = get_D("mnist", classes=11)
    torch.save(D.state_dict(), "test.pth")
    print(D(a))
