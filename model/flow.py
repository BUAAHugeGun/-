import torch
import numpy as np
import torch.nn as nn


def conv_layer(in_channels, out_channels, kernel, stride=1, padding=0, use_bn=True, bias=True):
    layers = []
    layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                            padding=padding, bias=bias))
    if use_bn:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def pool_layer(in_channels, kernel, stride, padding=0, mode="Max"):
    if mode != "Max" and mode != "Avg":
        assert 0
    if mode == "Max":
        return nn.Sequential(nn.MaxPool2d(kernel_size=kernel, stride=stride))
    else:
        return nn.Sequential(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))


class Permute(nn.Module):
    def __init__(self, n_channels, seed):
        super().__init__()

        permutation = np.arange(n_channels, dtype='int')
        np.random.seed(seed)
        np.random.shuffle(permutation)

        permutation_inv = np.zeros(n_channels, dtype='int')
        permutation_inv[permutation] = np.arange(n_channels, dtype='int')

        print('Permute: ', permutation, permutation_inv)
        self.permutation = torch.from_numpy(permutation).long()
        self.permutation_inv = torch.from_numpy(permutation_inv).long()

    def forward(self, x, ldj, reverse=False):
        print(self.permutation)
        if not reverse:
            x = x[:, self.permutation, :, :]
        else:
            x = x[:, self.permutation_inv, :, :]

        return x, ldj

    def InversePermute(self):
        inv_permute = Permute(len(self.permutation))
        inv_permute.permutation = self.permutation_inv
        inv_permute.permutation_inv = self.permutation
        return inv_permute


def space_to_depth(x):
    xs = x.size()
    # Pick off every second element
    x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
    # Transpose picked elements next to channels.
    x = x.permute((0, 1, 3, 5, 2, 4)).contiguous()
    # Combine with channels.
    x = x.view(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
    return x


def depth_to_space(x):
    xs = x.size()
    # Pick off elements from channels
    x = x.view(xs[0], xs[1] // 4, 2, 2, xs[2], xs[3])
    # Transpose picked elements next to HW dimensions.
    x = x.permute((0, 1, 4, 2, 5, 3)).contiguous()
    # Combine with HW dimensions.
    x = x.view(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
    return x


class Squeeze(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z = space_to_depth(z)
        else:
            z = depth_to_space(z)
        return z, ldj


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel, Conv2dAct):
        super().__init__()

        self.nn = torch.nn.Sequential(
            Conv2dAct(n_channels, n_channels, kernel, padding=1),
            torch.nn.Conv2d(n_channels, n_channels, kernel, padding=1),
        )

    def forward(self, x):
        h = self.nn(x)
        h = F.relu(h + x)
        return h


class DenseLayer(nn.Module):
    def __init__(self, n_inputs, growth, Conv2dAct):
        super().__init__()

        conv1x1 = Conv2dAct(
            n_inputs, n_inputs, kernel=1, stride=1,
            padding=0, bias=True)

        self.nn = torch.nn.Sequential(
            conv1x1,
            Conv2dAct(
                n_inputs, growth, kernel=3, stride=1,
                padding=1, bias=True),
        )

    def forward(self, x):
        h = self.nn(x)

        h = torch.cat([x, h], dim=1)
        return h


class DenseBlock(nn.Module):
    def __init__(
            self, n_inputs, n_outputs, kernel, Conv2dAct, densenet_depth, **kwargs):
        super().__init__()
        depth = densenet_depth

        future_growth = n_outputs - n_inputs

        layers = []

        for d in range(depth):
            growth = future_growth // (depth - d)

            layers.append(DenseLayer(n_inputs, growth, Conv2dAct))
            n_inputs += growth
            future_growth -= growth

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class NN(nn.Module):
    def __init__(
            self, c_in, c_out, nn_type, n_channels, kernel=3, **kwargs):
        super().__init__()

        if nn_type == 'resnet':
            layers = [
                conv_layer(c_in, n_channels, kernel, padding=1),
                ResidualBlock(n_channels, kernel, conv_layer),
                ResidualBlock(n_channels, kernel, conv_layer)]

            layers += [
                torch.nn.Conv2d(n_channels, c_out, kernel, padding=1)
            ]

        elif nn_type == 'densenet':
            layers = [
                DenseBlock(
                    n_inputs=c_in,
                    n_outputs=n_channels + c_in,
                    kernel=kernel,
                    Conv2dAct=conv_layer, **kwargs)]

            layers += [
                torch.nn.Conv2d(n_channels + c_in, c_out, kernel, padding=1)
            ]
        else:
            raise ValueError

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class Coupling(nn.Module):
    def __init__(self, c_in, factor, coupling_type, **kwargs):
        super().__init__()
        self.kernel = 3

        self.split_idx = c_in - (c_in // factor)

        self.nn = NN(
            c_in=self.split_idx,
            c_out=c_in - self.split_idx,
            kernel=self.kernel,
            nn_type=coupling_type, **kwargs)

    def forward(self, z, ldj, reverse=False):
        z1 = z[:, :self.split_idx, :, :]
        z2 = z[:, self.split_idx:, :, :]

        t = self.nn(z1)

        if not reverse:
            z2 = z2 + t
        else:
            z2 = z2 - t

        z = torch.cat([z1, z2], dim=1)

        return z, ldj


class FLOW(nn.Module):
    def __init__(self, n_levels, n_flows, splitfactor, **kwargs):
        super(FLOW, self).__init__()
        layers = []
        n_channels = 3
        layers.append(Squeeze())
        n_channels *= 4

        for level in range(n_levels):
            for i in range(n_flows):
                perm_layer = Permute(n_channels, i)
                layers.append(perm_layer)

                layers.append(
                    Coupling(n_channels, splitfactor, **kwargs))

            if level < n_levels - 1:
                layers.append(Squeeze())
                n_channels *= 4

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, z, reverse=False):
        ldj = torch.ones_like(z[:, 0, 0, 0])

        if not reverse:
            for l, layer in enumerate(self.layers):
                z, ldj = layer(z, ldj)

        else:
            for l, layer in reversed(list(enumerate(self.layers))):
                z, ldj = layer(z, ldj, reverse=True)

        return z, ldj


if __name__ == "__main__":
    flow = FLOW(2, 4, 2, coupling_type="densenet", n_channels=64, densenet_depth=6)
    x = torch.randn([8, 3, 32, 32])
    y = flow(x)
    print(y[0].shape)
    z = flow(y[0], reverse=True)
    print(z[0].shape)
    print((z[0] - x).sum())
