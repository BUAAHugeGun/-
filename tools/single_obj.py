from dataset.data_builder import build_data
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from loss import SSIM_Loss
import torch.nn as nn
from nets.generator import get_G
from nets.discriminator import get_D
from torch import autograd
from tqdm import tqdm
from model.flow import GLOW
import math
import cv2
import torch
import numpy as np
import yaml
import os
import argparse
from tools.coco_cut import classes as coco_classes


def to_log(s, output=True):
    global log_file
    if output:
        print(s)
    print(s, file=log_file)


def open_config(root):
    f = open(os.path.join(root, "config.yaml"))
    config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load(models, epoch, root):
    def _detect_latest():
        checkpoints = os.listdir(os.path.join(root, "logs"))
        checkpoints = [f for f in checkpoints if f.startswith("G_epoch-") and f.endswith(".pth")]
        checkpoints = [int(f[len("G_epoch-"):-len(".pth")]) for f in checkpoints]
        checkpoints = sorted(checkpoints)
        _epoch = checkpoints[-1] if len(checkpoints) > 0 else None
        return _epoch

    if epoch == -1:
        epoch = _detect_latest()
    if epoch is None:
        return -1
    for name, model in models.items():
        ckpt = torch.load(os.path.join(root, "logs/" + name + "_epoch-{}.pth".format(epoch)))
        ckpt = {k: v for k, v in ckpt.items()}
        model.load_state_dict(ckpt)
        print("load model: {} from epoch: {}".format(name, epoch))
    return epoch


def make_noise(bs, noise_dim):
    if noise_dim == 0:
        return None
    noise = torch.randn([bs, noise_dim]).cuda()
    return noise


class SingleObj():
    def __init__(self, args, root):
        if args['classes'] == 'NONE':
            args['classes'] = list(coco_classes.keys())
        self.classes_num = len(args['classes'])
        self.noise_dim = args['noise_dim'] if self.classes_num > 1 else 0

        self.G = get_G("unet", in_channels=1, out_channels=3, scale=6, noise_dim=self.noise_dim,
                       image_size=args['image_size'], classes_num=self.classes_num).cuda()
        self.D = get_D("dnn", classes=self.classes_num + 1).cuda()

        load({"G": self.G}, args["load_epoch"], root)

        self.G.eval()
        print("object generator ready!")

    def generate(self, mask, labels):
        noise = make_noise(mask.shape[0], self.noise_dim)
        with torch.no_grad():
            G_out = self.G(mask, noise, labels)
        return G_out
