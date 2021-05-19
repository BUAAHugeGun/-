from dataset.data_builder import build_data
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
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
from loss.PSNR_Loss import Loss as PNSR
from loss.SSIM_Loss import MSSSIM
from tools.coco_cut import classes as coco_classes
from tools.single_obj import SingleObj

log_file = None


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
    # print("loaded from epoch: {}".format(epoch))
    return epoch


def calc_gradient_penalty(netD, origin, fake_data, batch_size, gp_lambda):
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(origin.shape).contiguous()
    alpha = alpha.cuda()

    interpolates = alpha * origin + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates.requires_grad = True

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


# BCHW
def batch_image_merge(image):
    # image = torch.cat(image.split(4, 0), 2)
    image = torch.cat(image.split(1, 0), 3)
    # CH'W'
    return image


def imagetensor2np(x):
    x = torch.round((x + 1) / 2 * 255).clamp(0, 255).int().abs()
    x = x.detach().cpu().numpy()
    x = np.array(x, dtype=np.uint8).squeeze(0)
    x = np.transpose(x, [1, 2, 0])
    return x


def test(args, root):
    print(args)
    if not os.path.exists(os.path.join(root, "test")):
        os.mkdir(os.path.join(root, "test"))
    if args['classes'] == 'NONE':
        args['classes'] = list(coco_classes.keys())
    classes_num = len(args['classes'])

    if classes_num == 1:
        single_root = '../experiments/pix2pix_person'
    elif classes_num == 5:
        single_root = '../experiments/pix2pix_5class_new_nfl'
    # elif classes_num == 10:
    #    single_root = '../experiments/p2p_10class'
    else:
        assert 0
    single_model = SingleObj(open_config(single_root), single_root)
    data_root = os.path.join(args['data_path'], "COCO", "results_coco_val_{}".format(classes_num))
    from dataset.data_builder import coco_synthesis_dataset
    dataset = coco_synthesis_dataset(data_root, False, classes=args['classes'], image_size=args['image_size'],
                                     obj_model=single_model)

    #G = get_G("mini").cuda()
    G = get_G("post", in_channels=3, out_channels=3, scale=5, image_size=args['image_size']).cuda()
    G.eval()

    psnr = PNSR()
    msssim = MSSSIM()
    load({"G": G}, args["load_epoch"], root)
    l1_sum = 0
    psnr_sum = 0
    ms_ssim_sum = 0
    with torch.no_grad():
        for i in tqdm(range(len(dataset))):
            filename = dataset.image_id_to_file_name[i]
            synthesis, origin, shape = dataset[i]
            synthesis, origin = synthesis.cuda().unsqueeze(0), origin.cuda().unsqueeze(0)

            # G
            G_out = G(synthesis)

            G_out = G_out / 2 + 0.5
            G_out = G_out.clamp(0, 1)
            origin = origin / 2 + 0.5

            l1 = nn.L1Loss()(G_out, origin)
            psnr_ = psnr(G_out, origin)
            ms_ssim_ = msssim(G_out, origin)
            l1_sum += l1
            psnr_sum += psnr_
            ms_ssim_sum += ms_ssim_

            save_image(G_out, os.path.join(root, "test", filename + ".png"))
            # save_image(origin, os.path.join(root, "test", filename + "_ori.png"))
    length = len(dataset)
    print(l1_sum.item() / length, psnr_sum.item() / length, ms_ssim_sum.item() / length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()
    test(open_config(args.root), args.root)
