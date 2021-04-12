import torch
from torch import nn
from torchvision.models import resnet50
from torchvision.models import resnet34
from torchvision.models import resnet101


def _conv_layer(in_channels, out_channels, kernel, stride, padding, bias=True, norm=True, act="relu"):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                  padding=padding, bias=bias)
    ]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if act == "relu":
        layers.append(nn.LeakyReLU(0.2))
    elif act == "tanh":
        layers.append(nn.Tanh())
    elif act == "sigmoid":
        layers.append(nn.Sigmoid())
    elif act == "NONE":
        pass
    else:
        assert 0
    return nn.Sequential(*layers)


def _deconv_layer(in_channels, out_channels, kernel, stride, padding, bias=True, norm=True):
    layers = [
        nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                  padding=padding, bias=bias),
    ]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _pool_layer(kernel, stride, padding, mode="max"):
    if mode == "max":
        return nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
    elif mode == "avg":
        return nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)
    else:
        print("wrong pool layer mode: {}".format(mode))
        assert 0


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
        self.conv1 = nn.Conv2d(6, 6, 3, 1, 1)
        self.conv2 = nn.Conv2d(6, 3, 3, 1, 1)
        self.model = resnet34()
        self.classes = classes
        self.validity_layer = nn.Sequential(nn.Linear(1000, 1), nn.Sigmoid())
        self.label_layer = nn.Sequential(nn.Linear(1000, self.classes), nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.model(self.conv2(self.conv1(x)))
        validity = self.validity_layer(x).view(-1)
        plabel = self.label_layer(x).view(-1, self.classes)
        return validity, plabel


class Discriminator(nn.Module):
    def __init__(self, classes):
        super(Discriminator, self).__init__()
        self.classes = classes
        self.conv = nn.Sequential(
            # 6,256,256
            _conv_layer(4, 64, 4, 2, 1, norm=False),
            # _pool_layer(2, 2, 0),
            _conv_layer(64, 128, 4, 2, 1, bias=False),
            # _pool_layer(2, 2, 0),
            # 128,64,64
            _conv_layer(128, 256, 4, 2, 1, bias=False),
            # _pool_layer(2, 2, 0),
            _conv_layer(256, 512, 4, 1, 1, bias=False),
            # 512,32,32
            _conv_layer(512, 1, 4, 1, 1, norm=False),
            # _conv_layer(128, 1, 3, 1, 1, bias=False),
            # 1,32,32
        )
        self.validity_layer = nn.Sequential(nn.Linear(100, 1))
        self.label_layer = nn.Sequential(nn.Linear(100, self.classes), nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.conv(x)
        return x, 0


def get_D(tag, **kwargs):
    if tag == "mnist":
        return MNIST_D()
    elif tag == "resnet":
        classes = kwargs.get("classes", None)
        if classes is None:
            print("resnet need parameter: classes")
            assert 0
        return Resnet(classes)
    elif tag == 'dnn':
        classes = kwargs.get("classes", None)
        if classes is None:
            print("dnn need parameter: classes")
            assert 0
        return Discriminator(classes)


if __name__ == "__main__":
    a = torch.randn([8, 6, 256, 256])
    D = get_D("dnn", classes=2)
    torch.save(D.state_dict(), "test.pth")
    print(D.conv)
