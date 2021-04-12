import torch
from torch.nn import Module, Conv2d
from torch.nn.utils import spectral_norm
from torch.nn.functional import interpolate, relu
from torch import nn


class SPADE(nn.Module):
    # seg_channel : # channel of segmentation map
    # main_channel : # channel of main input and output stream channel
    def __init__(self, main_channel):
        super(SPADE, self).__init__()
        self.seg_channel = 1
        self.main_channel = main_channel
        self.n_hidden = 128

        # self.batch = nn.SyncBatchNorm(self.main_channel)
        self.batch = nn.BatchNorm2d(self.main_channel)

        self.share_cov = nn.Sequential(
            nn.Conv2d(in_channels=self.seg_channel, out_channels=self.n_hidden, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.gamma = nn.Conv2d(in_channels=self.n_hidden, out_channels=self.main_channel, kernel_size=3, stride=1,
                               padding=1)
        self.beta = nn.Conv2d(in_channels=self.n_hidden, out_channels=self.main_channel, kernel_size=3, stride=1,
                              padding=1)

    def forward(self, x, seg):
        x = self.batch(x)  # input channel

        seg = interpolate(input=seg, size=x.shape[2:], mode='nearest')
        seg_share = self.share_cov(seg)
        seg_gamma = self.gamma(seg_share)
        seg_beta = self.beta(seg_share)

        x = x * (1 + seg_gamma) + seg_beta

        return x


class SPADE_CONV(Module):
    def __init__(self, conv_layer, in_channels, out_channels, kernel, stride, padding, bias=True, norm=True,
                 act='relu'):
        super(SPADE_CONV, self).__init__()
        self.conv = conv_layer(in_channels, out_channels, kernel, stride, padding, bias=bias)
        if norm:
            self.norm = SPADE(out_channels)
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
            x = self.norm(x, seg)
        return self.act(x)


class SPADE_POOL(Module):
    def __init__(self, pool_layer, kernel, stride, padding):
        super(SPADE_POOL, self).__init__()
        self.pool = pool_layer(kernel, stride, padding)

    def forward(self, x, seg):
        return self.pool(x)


if __name__ == "__main__":
    spade = SPADE_CONV(nn.Conv2d, 1, 64, 3, 1, 1)
    seg = torch.randn(32, 1, 64, 64)
    a = torch.randn([32, 1, 64, 64])
    num_params = 0
    for param in spade.parameters():
        num_params += param.numel()
    print(spade(a, seg).shape)
    print(num_params / 1e6)
    print(spade)
