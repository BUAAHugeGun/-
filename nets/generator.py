import torch
from torch import nn
import math
from nets.spade import SPADE, SPADE_CONV, SPADE_POOL


class MNIST_G(nn.Module):
    def __init__(self):
        super(MNIST_G, self).__init__()
        layers = [
            nn.Conv2d(2, 4, 5, 1, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, 5, 1, 2),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, 5, 1, 2),
            nn.Tanh(),
        ]
        self.conv = nn.Sequential(*layers)
        """
        self.fc = nn.Linear(100, 3136)  # batch, 3136=1x56x56
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
            nn.Tanh()
        )
        """

    def forward(self, x):
        return self.conv(x)
        """
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x
        """


def _conv_layer(in_channels, out_channels, kernel, stride, padding, bias=True, norm=True, act="relu"):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                  padding=padding, bias=bias)
    ]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    if act == "relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    elif act == "tanh":
        layers.append(nn.Tanh())
    elif act == "sigmoid":
        layers.append(nn.Sigmoid())
    else:
        assert 0
    return nn.Sequential(*layers)


def _deconv_layer(in_channels, out_channels, kernel, stride, padding, bias=True, norm=True):
    layers = [
        # nn.UpsamplingBilinear2d(scale_factor=2),
        # nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
        #          padding=padding, bias=bias),
        nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
    ]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def _pool_layer(kernel, stride, padding, mode="max"):
    if mode == "max":
        return nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
    elif mode == "avg":
        return nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)
    else:
        print("wrong pool layer mode: {}".format(mode))
        assert 0


class UNET_BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, up=False, norm1=True, norm2=True, half=True):
        super(UNET_BLOCK, self).__init__()
        half = up and half
        if pool:
            self.pool = nn.AvgPool2d(2, 2, 0)
        else:
            self.pool = None
        # layers.append(_conv_layer(in_channels, out_channels, 3, 1, 1, norm=norm1))
        if up:
            if half:
                self.layer = SPADE_CONV(nn.ConvTranspose2d, in_channels, out_channels // 2, 4, 2, 1, norm=norm2)
            else:
                self.layer = SPADE_CONV(nn.ConvTranspose2d, in_channels, out_channels, 4, 2, 1, norm=norm2)
        else:
            self.layer = SPADE_CONV(nn.Conv2d, in_channels, out_channels, 3, 1, 1, norm=norm2)

    def forward(self, x, seg):
        if self.pool is not None:
            x = self.pool(x)
        return self.layer(x, seg)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, scale=5, Max=512):
        super(UNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.Max = Max
        self.build()

    def initial(self, scale_factor=1.0, mode="FAN_IN"):
        if mode != "FAN_IN" and mode != "FAN_out":
            assert 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if mode == "FAN_IN":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(scale_factor / n))
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(scale_factor / n))
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def build(self):
        self.G = []
        self.D = []
        self.pre_conv = SPADE_CONV(nn.Conv2d, self.in_channels, 32, 3, 1, 1, norm=False)
        for i in range(self.scale):
            in_channels = 32 * (2 ** i)
            self.G.append(
                UNET_BLOCK(min(self.Max, in_channels), min(self.Max, in_channels * 2), True, i == (self.scale - 1),
                           norm1=(i > 0), norm2=i < self.scale, half=False))
        for i in range(self.scale):
            in_channels = 32 * (2 ** (self.scale - i))
            out_channels = in_channels // 2
            self.D.append(
                UNET_BLOCK(min(self.Max * 2, in_channels), min(self.Max * 2, out_channels), False,
                           i < (self.scale - 1)))
        self.post_conv = SPADE_CONV(nn.Conv2d, 32, self.out_channels, 3, 1, 1, act="tanh", norm=False)
        self.g_list = nn.Sequential(*self.G)
        self.d_list = nn.Sequential(*self.D)

    def forward(self, x):
        seg = x.clone().detach()
        out = []
        out.append(self.pre_conv(x, seg))
        for i in range(self.scale):
            out.append(self.G[i](out[i], seg))
        for i in range(self.scale):
            j = self.scale - i - 1
            input = torch.cat([out[j + 1], out[j]], 1)
            out[j] = self.D[i](input, seg)
        return self.post_conv(out[0], seg)


def get_G(tag, **kwargs):
    if tag == "mnist":
        return MNIST_G()
    if tag == "unet":
        in_channels = kwargs.get("in_channels", None)
        out_channels = kwargs.get("out_channels", None)
        scale = kwargs.get("scale", 5)
        if in_channels is not None and out_channels is not None:
            return UNET(in_channels, out_channels, scale)
        else:
            print("unet need parameter: in_channels or outchannels")
            assert 0


if __name__ == "__main__":
    a = torch.randn([32, 1, 64, 64])
    aa = a.clone()
    a.requires_grad = True
    G = get_G("unet", in_channels=1, out_channels=3, scale=6)
    # print(G)
    print(G(a).shape)
    num_params = 0
    for param in G.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
