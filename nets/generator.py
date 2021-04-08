import torch
from torch import nn
import math


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

    def get_block(self, in_channels, out_channels, pool=True, up=False, norm1=True, norm2=True):
        layers = []
        if pool:
            layers.append(_pool_layer(2, 2, 0, "avg"))
        layers.append(_conv_layer(in_channels, out_channels, 3, 1, 1, norm=norm1))
        if up:
            layers.append(_deconv_layer(out_channels, out_channels // 2, 3, 1, 1, norm=norm2))
        else:
            layers.append(_conv_layer(out_channels, out_channels, 3, 1, 1, norm=norm2))
        return nn.Sequential(*layers)

    def build(self):
        self.G = []
        self.D = []
        self.pre_conv = _conv_layer(self.in_channels, 32, 3, 1, 1, norm=False)
        for i in range(self.scale):
            in_channels = 32 * (2 ** i)
            self.G.append(
                self.get_block(min(self.Max, in_channels), min(self.Max, in_channels * 2), True, i == (self.scale - 1),
                               norm1=(i > 0), norm2=i < self.scale))
        for i in range(self.scale):
            in_channels = 32 * (2 ** (self.scale - i))
            out_channels = in_channels // 2
            self.D.append(
                self.get_block(min(self.Max, in_channels), min(self.Max, in_channels), False, i < (self.scale - 1)))
        self.post_conv = _conv_layer(32, self.out_channels, 3, 1, 1, act="tanh", norm=False)
        self.g_list = nn.Sequential(*self.G)
        self.d_list = nn.Sequential(*self.D)

    def forward(self, x):
        out = []
        out.append(self.pre_conv(x))
        for i in range(self.scale):
            out.append(self.G[i](out[i]))
        for i in range(self.scale):
            j = self.scale - i - 1
            input = torch.cat([out[j + 1], out[j]], 1)
            out[j] = self.D[i](input)
        return self.post_conv(out[0])


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


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels,
                              64,
                              normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512, normalize=False)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


if __name__ == "__main__":
    a = torch.randn([4, 2, 64, 64])
    aa = a.clone()
    a.requires_grad = True
    G = get_G("unet", in_channels=2, out_channels=1, scale=6)
    print(G)
    num_params = 0
    for param in G.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
