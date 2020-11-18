from dataset.dataset import build_data
from nets.generator import get_G
from nets.discriminator import get_D

import yaml
import os
import argparse


def open_config(root):
    f = open(os.path.join(root, "config.yaml"))
    config = yaml.load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--test", default=False, action='store_true')
    args = parser.parse_args()
    print(args)
    args = open_config(args.root)
    print(args)
