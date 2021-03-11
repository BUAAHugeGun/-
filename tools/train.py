from dataset.data_builder import build_data
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import torch.nn as nn
from nets.generator import get_G
from nets.discriminator import get_D
from tqdm import tqdm
from model.flow import GLOW
import math
import torch

import yaml
import os
import argparse

log_file = None


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
        to_log("load model: {} from epoch: {}".format(name, epoch))
    # print("loaded from epoch: {}".format(epoch))
    return epoch


def calc_z_shapes(n_channel, input_size, n_levels):
    z_shapes = []

    for i in range(n_levels - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, image_size, n_bits):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -n_bits * n_pixel
    loss = loss + logdet + log_p

    return (
        (-loss / (math.log(2) * n_pixel)).mean(),
        (log_p / (math.log(2) * n_pixel)).mean(),
        (logdet / (math.log(2) * n_pixel)).mean(),
    )


def train(args, root):
    image_size = 256
    global log_file
    if not os.path.exists(os.path.join(root, "logs")):
        os.mkdir(os.path.join(root, "logs"))
    if not os.path.exists(os.path.join(root, "logs/result/")):
        os.mkdir(os.path.join(root, "logs/result/"))
    if not os.path.exists(os.path.join(root, "logs/result/event")):
        os.mkdir(os.path.join(root, "logs/result/event"))
    log_file = open(os.path.join(root, "logs/log.txt"), "w")
    to_log(args)
    writer = SummaryWriter(os.path.join(root, "logs/result/event/"))

    dataloader = build_data(args['data_tag'], args['data_path'], args["bs"], True, num_worker=args["num_workers"])

    G = get_G("unet", in_channels=3, out_channels=3, scale=5).cuda()
    D = get_D("resnet", classes=2).cuda()

    g_opt = torch.optim.Adam(G.parameters(), lr=args["lr"])
    d_opt = torch.optim.Adam(D.parameters(), lr=args["lr"])
    g_sch = torch.optim.lr_scheduler.MultiStepLR(g_opt, args["lr_milestone"], gamma=0.5)
    d_sch = torch.optim.lr_scheduler.MultiStepLR(d_opt, args["lr_milestone"], gamma=0.5)

    load_epoch = load({"G": G, "D": D, "g_opt": g_opt, "d_opt": d_opt, "g_sch": g_sch, "d_sch": d_sch},
                      args["load_epoch"], root)
    tot_iter = (load_epoch + 1) * len(dataloader)

    validity_loss = nn.BCELoss().cuda()

    real_label = torch.ones([1]).cuda()
    fake_label = torch.zeros([1]).cuda()

    g_opt.step()
    d_opt.step()
    for epoch in range(load_epoch + 1, args['epoch']):
        g_sch.step()
        d_sch.step()
        for i, (image, mask) in enumerate(dataloader):
            tot_iter += 1
            image, mask = image.cuda(), mask.cuda()

            d_opt.zero_grad()
            # D_real
            validity_label = real_label.expand(image.shape[0])
            pvalidity, plabels = D(image)
            D_loss_real_val = validity_loss(pvalidity, validity_label)

            D_loss_real = D_loss_real_val
            D_loss_real.backward()
            # D_r = pvalidity.mean()

            # D_fake
            G_out = G(mask)
            validity_label = fake_label.expand(image.shape[0])
            pvalidity, plabels = D(G_out.detach())
            D_loss_fake_val = validity_loss(pvalidity, validity_label)

            D_loss_fake = D_loss_fake_val
            D_loss_fake.backward()
            # D_f = pvalidity.mean()

            D_loss = D_loss_real + D_loss_fake
            d_opt.step()

            g_opt.zero_grad()
            # G
            # input = torch.randn([image.shape[0], 2, image_size, image_size]).cuda()
            # input[:, 1, :, :] = label.reshape(image.shape[0], 1, 1).expand(image.shape[0], image_size, image_size)
            validity_label = real_label.expand(image.shape[0])
            G_out = G(mask)
            pvalidity, plabels = D(G_out)
            G_loss_val = validity_loss(pvalidity, validity_label)

            G_loss = G_loss_val
            G_loss.backward()

            # DG_r = pvalidity.mean()
            g_opt.step()

            if tot_iter % args['show_interval'] == 0:
                to_log(
                    'epoch: {}, batch: {}, D_loss: {:.5f}, D_loss_real: {:.5f}, D_loss_fake: {:.5f}, D_loss_real_val: {:.5f}, D_loss_fake_val: {:.5f}, G_loss: {:5f}, G_loss_val: {:.5f}, lr: {:.5f}'.format(
                        epoch, i, D_loss.item(), D_loss_real.item(), D_loss_fake.item(), D_loss_real_val.item(),
                        D_loss_fake_val.item(), G_loss.item(),G_loss_val.item(), g_sch.get_last_lr()[0]))
                writer.add_scalar("loss/D_loss", D_loss.item(), tot_iter)
                writer.add_scalar("loss/D_loss_real", D_loss_real.item(), tot_iter)
                writer.add_scalar("loss/D_loss_fake", D_loss_fake.item(), tot_iter)
                writer.add_scalar("loss/D_loss_real_val", D_loss_real_val.item(), tot_iter)
                writer.add_scalar("loss/D_loss_fake_val", D_loss_fake_val.item(), tot_iter)
                writer.add_scalar("loss/G_loss", G_loss.item(), tot_iter)
                writer.add_scalar("loss/G_loss_val", G_loss_val.item(), tot_iter)
                writer.add_scalar("lr", g_sch.get_last_lr()[0], tot_iter)

        if epoch % args["snapshot_interval"] == 0:
            torch.save(G.state_dict(), os.path.join(root, "logs/G_epoch-{}.pth".format(epoch)))
            torch.save(D.state_dict(), os.path.join(root, "logs/D_epoch-{}.pth".format(epoch)))
            torch.save(g_opt.state_dict(), os.path.join(root, "logs/g_opt_epoch-{}.pth".format(epoch)))
            torch.save(d_opt.state_dict(), os.path.join(root, "logs/d_opt_epoch-{}.pth".format(epoch)))
            torch.save(g_sch.state_dict(), os.path.join(root, "logs/g_sch_epoch-{}.pth".format(epoch)))
            torch.save(d_sch.state_dict(), os.path.join(root, "logs/d_sch_epoch-{}.pth".format(epoch)))
        if epoch % args['test_interval'] == 0:
            # label = torch.tensor([5]).expand([64]).cuda()
            # input_test[:, 1, :, :] = label.reshape(64, 1, 1).expand(64, image_size, image_size)
            # G_out = G(input_test)
            G_out = G_out / 2 + 0.5
            save_image(G_out, os.path.join(root, "logs/output-{}.png".format(epoch)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--test", default=False, action='store_true')
    args = parser.parse_args()
    train(open_config(args.root), args.root)
