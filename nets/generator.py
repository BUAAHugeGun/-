import torch
from torch import nn
import math
from nets.spade import SPADE, SPADE_CONV, SPADE_POOL, _CONV


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
    def __init__(self, in_channels, out_channels, scale=5, Max=512, noise_dim=100, image_size=64, classes_num=5):
        super(UNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = scale
        self.Max = Max
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.classes_num = classes_num
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
        if self.noise_dim > 0:
            self.nosie_fc = nn.Linear(self.noise_dim, self.image_size * self.image_size)
            self.embedding = nn.Embedding(self.classes_num, self.noise_dim)
            self.in_channels += 1
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

    def forward(self, x, noise, label):
        seg = x.clone().detach()
        if noise is not None:
            label_embedding = self.embedding(label)
            noise = torch.mul(noise, label_embedding)
            noise = self.nosie_fc(noise)
            noise = torch.reshape(noise, [-1, 1, self.image_size, self.image_size])
            x = torch.cat([x, noise], 1)
        out = []
        out.append(self.pre_conv(x, seg))
        for i in range(self.scale):
            out.append(self.G[i](out[i], seg))
        for i in range(self.scale):
            j = self.scale - i - 1
            input = torch.cat([out[j + 1], out[j]], 1)
            out[j] = self.D[i](input, seg)
        return self.post_conv(out[0], seg)


class _BLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True, up=False, norm1=True, norm2=True, half=True):
        super(_BLOCK, self).__init__()
        half = up and half
        if pool:
            self.pool = nn.AvgPool2d(2, 2, 0)
        else:
            self.pool = None
        # layers.append(_conv_layer(in_channels, out_channels, 3, 1, 1, norm=norm1))
        if up:
            if half:
                self.layer = _CONV(nn.ConvTranspose2d, in_channels, out_channels // 2, 4, 2, 1, norm=norm2)
            else:
                self.layer = _CONV(nn.ConvTranspose2d, in_channels, out_channels, 4, 2, 1, norm=norm2)
        else:
            self.layer = _CONV(nn.Conv2d, in_channels, out_channels, 3, 1, 1, norm=norm2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.layer(x)


class POST(nn.Module):
    def __init__(self, in_channels, out_channels, scale=5, Max=512, ):
        super(POST, self).__init__()
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
        self.pre_conv = _CONV(nn.Conv2d, self.in_channels, 32, 3, 1, 1, norm=False)
        for i in range(self.scale):
            in_channels = 32 * (2 ** i)
            self.G.append(
                _BLOCK(min(self.Max, in_channels), min(self.Max, in_channels * 2), True, i == (self.scale - 1),
                       norm1=(i > 0), norm2=i < self.scale, half=False))
        for i in range(self.scale):
            in_channels = 32 * (2 ** (self.scale - i))
            out_channels = in_channels // 2
            self.D.append(
                _BLOCK(min(self.Max * 2, in_channels), min(self.Max * 2, out_channels), False,
                       i < (self.scale - 1)))
        self.post_conv = _CONV(nn.Conv2d, 32, self.out_channels, 3, 1, 1, act="tanh", norm=False)
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

class mini(nn.Module):
    def __init__(self, in_channels=3):
        super(mini, self).__init__()
        self.in_channels = in_channels
        self.build()

    def _conv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=False):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                                padding=padding, bias=bias, padding_mode='reflect'))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.0))
        return nn.Sequential(*layers)

    def _deconv_layer(self, in_channels, out_channels, kernel, stride, padding, bias=True, bn=False):
        layers = []
        layers.append(nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
            padding=padding, bias=bias))
        if bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.0))
        return nn.Sequential(*layers)

    def _pool_layer(self, kernel, stride, padding=0, mode="Max"):
        if mode != "Max" and mode != "Avg":
            assert 0
        if mode == "Max":
            return nn.Sequential(nn.MaxPool2d(kernel_size=kernel, stride=stride))
        else:
            return nn.Sequential(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))

    def build_layers(self, in_channels, out_channels):
        layers = []
        layers.append(self._conv_layer(in_channels, out_channels, 3, 1, 1))
        layers.append(self._conv_layer(out_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)

    def build(self):
        self.f0 = []
        self.f1 = []
        self.g0 = []
        self.g1 = []
        down_channels = [self.in_channels, 16, 32, 64, 128, 256]
        up_channels = [256, 128, 64, 32, 16]
        for i in range(5):
            self.f0.append(self.build_layers(down_channels[i], down_channels[i + 1]))
            self.f1.append(self._pool_layer(2, 2))
        for i in range(4):
            self.g0.append(self._deconv_layer(up_channels[i], up_channels[i + 1], 4, 2, 1))
            self.g1.append(self.build_layers(up_channels[i], up_channels[i + 1]))

        self.G = self._conv_layer(up_channels[4], self.in_channels, 3, 1, 1)
        self.f0_list = nn.Sequential(*self.f0)
        self.f1_list = nn.Sequential(*self.f1)
        self.g0_list = nn.Sequential(*self.g0)
        self.g1_list = nn.Sequential(*self.g1)

    def forward(self, x):
        x_ori = x.clone()
        x_down = []
        x_down.append(x)
        for i in range(5):
            x_down.append(self.f0[i](x))
            x = self.f1[i](x_down[i + 1])
        x = x_down[5]
        for i in range(4):
            x = self.g0[i](x)
            x = torch.cat([x, x_down[4 - i]], 1)
            x = self.g1[i](x)
        return self.G(x) + x_ori


def get_G(tag, **kwargs):
    if tag == "mnist":
        return MNIST_G()
    if tag == "unet":
        in_channels = kwargs.get("in_channels", None)
        out_channels = kwargs.get("out_channels", None)
        noise_dim = kwargs.get("noise_dim", None)
        image_size = kwargs.get("image_size", None)
        scale = kwargs.get("scale", None)
        classes_num = kwargs.get("classes_num", None)
        if in_channels is not None and out_channels is not None:
            return UNET(in_channels, out_channels, scale, noise_dim=noise_dim, image_size=image_size,
                        classes_num=classes_num)
        else:
            print("unet need parameter: in_channels or outchannels")
            assert 0
    if tag == 'post':
        in_channels = kwargs.get("in_channels", None)
        out_channels = kwargs.get("out_channels", None)
        scale = kwargs.get("scale", None)
        return POST(in_channels, out_channels, scale)
    if tag=="mini":
        return mini()


if __name__ == "__main__":
    a = torch.randn([16, 3, 64, 64])
    label = torch.randint(0, 5, a.shape[0:1])
    print(label)
    noise = torch.randn(16, 100)
    a.requires_grad = True
    G = get_G("post", in_channels=3, out_channels=3, scale=5)
    print(G(a).shape)
    num_params = 0
    for param in G.parameters():
        num_params += param.numel()
    print(num_params / 1e6)
    torch.save(G.state_dict(), "test.pth")
