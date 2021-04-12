import torch
from torch.nn import Module, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import interpolate, relu
from torch import nn


class SPADE(Module):
    def __init__(self, k, spade_filter=128, spade_kernel=3):
        super().__init__()
        num_filters = spade_filter
        kernel_size = spade_kernel
        self.conv = spectral_norm(Conv2d(1, num_filters, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_gamma = spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))
        self.conv_beta = spectral_norm(Conv2d(num_filters, k, kernel_size=(kernel_size, kernel_size), padding=1))

    def forward(self, x, seg):
        N, C, H, W = x.size()

        sum_channel = torch.sum(x.reshape(N, C, H * W), dim=-1)
        mean = sum_channel / (N * H * W)
        std = torch.sqrt((sum_channel ** 2 - mean ** 2) / (N * H * W))

        mean = torch.unsqueeze(torch.unsqueeze(mean, -1), -1)
        std = torch.unsqueeze(torch.unsqueeze(std, -1), -1)
        x = (x - mean) / std

        seg = interpolate(seg, size=(H, W), mode='nearest')
        seg = relu(self.conv(seg))
        seg_gamma = self.conv_gamma(seg)
        seg_beta = self.conv_beta(seg)

        x = torch.matmul(seg_gamma, x) + seg_beta

        return x


class SPADE_CONV(Module):
    def __init__(self, conv_layer, in_channels, out_channels, kernel, stride, padding, bias=True, norm=True,
                 act='relu'):
        super(SPADE_CONV, self).__init__()
        self.conv = conv_layer(in_channels, out_channels, kernel, stride, padding, bias=bias)
        if norm:
            self.norm = nn.InstanceNorm2d(out_channels)  # SPADE(out_channels)
        else:
            self.norm = None
        if act == "relu":
            self.act = nn.LeakyReLU(0.2, inplace=True)
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            assert 0

    def forward(self, x, seg):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)  # , seg)
        return self.act(x)


class SPADE_POOL(Module):
    def __init__(self, pool_layer, kernel, stride, padding):
        super(SPADE_POOL, self).__init__()
        self.pool = pool_layer(kernel, stride, padding)

    def forward(self, x, seg):
        return self.pool(x)


if __name__ == "__main__":
    spade = SPADE_CONV(nn.Conv2d, 1, 64, 5, 2, 1)
    seg = torch.randn(32, 1, 64, 64)
    a = torch.randn([32, 1, 64, 64])
    num_params = 0
    for param in spade.parameters():
        num_params += param.numel()
    print(spade(a, seg).shape)
    print(num_params / 1e6)
    print(spade)
