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
from tools.network import define_G, define_D, GANLoss, get_scheduler, update_learning_rate, NLayerDiscriminator

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


def calc_gradient_penalty(netD, real_data, label, fake_data, batch_size, gp_lambda):
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand(real_data.shape).contiguous()
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates.requires_grad = True

    disc_interpolates = netD(torch.cat([label, interpolates], 1))[0]

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


def train(args, root):
    image_size = 128
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

    dataloader = build_data(args['data_tag'], args['data_path'], args["bs"], True, num_worker=args["num_workers"],
                            classes=[1])

    G = get_G("unet", in_channels=1, out_channels=3, scale=5).cuda()
    D = get_D("dnn", classes=2).cuda()  # NLayerDiscriminator(6).cuda()# define_D(3 + 3, 64, 'basic', gpu_id=device) #

    g_opt = torch.optim.Adam(G.parameters(), lr=args["lr"], betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args["lr"], betas=(0.5, 0.999))
    g_sch = torch.optim.lr_scheduler.MultiStepLR(g_opt, args["lr_milestone"], gamma=0.5)
    d_sch = torch.optim.lr_scheduler.MultiStepLR(d_opt, args["lr_milestone"], gamma=0.5)

    load_epoch = load({"G": G, "D": D, "g_opt": g_opt, "d_opt": d_opt, "g_sch": g_sch, "d_sch": d_sch},
                      args["load_epoch"], root)
    tot_iter = (load_epoch + 1) * len(dataloader)

    validity_loss = nn.MSELoss().cuda()
    msssim = SSIM_Loss.msssim

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

            for _ in range(0, args['D_iter']):
                d_opt.zero_grad()
                # D_real
                pvalidity, plabels = D(torch.cat([mask, image], 1))
                validity_label = real_label.expand(pvalidity.shape)
                D_loss_real_val = validity_loss(pvalidity, validity_label)

                D_loss_real = D_loss_real_val

                # D_fake
                G_out = G(mask)
                pvalidity, plabels = D(torch.cat([mask, G_out.detach()], 1))
                validity_label = fake_label.expand(pvalidity.shape)
                D_loss_fake_val = validity_loss(pvalidity, validity_label)

                D_loss_fake = D_loss_fake_val

                D_loss = (D_loss_real + D_loss_fake) / 2
                D_loss.backward()

                # wgan-gp
                gradient_penalty = calc_gradient_penalty(D, image, mask, G_out.detach(), mask.shape[0],
                                                         args['gp_lambda'])
                gradient_penalty.backward()
                d_opt.step()

            g_opt.zero_grad()
            # G
            # input = torch.randn([image.shape[0], 2, image_size, image_size]).cuda()
            # input[:, 1, :, :] = label.reshape(image.shape[0], 1, 1).expand(image.shape[0], image_size, image_size)
            G_out = G(mask)
            pvalidity, plabels = D(torch.cat([mask, G_out], 1))
            validity_label = real_label.expand(pvalidity.shape)
            l1_loss = nn.L1Loss()(G_out, image)
            msssim_loss = 1 - msssim(G_out, image, normalize=True)
            G_loss_val = validity_loss(pvalidity, validity_label)

            G_loss = G_loss_val + l1_loss * args['lambda_l1'] + msssim_loss * args['lambda_ssim']
            G_loss.backward()

            # DG_r = pvalidity.mean()
            g_opt.step()

            if tot_iter % args['show_interval'] == 0:
                to_log(
                    'epoch: {}, batch: {}, D_loss: {:.5f}, D_loss_real: {:.5f}, D_loss_fake: {:.5f},' \
                    'D_loss_real_val: {:.5f},\ D_loss_fake_val: {:.5f}, G_loss: {:5f}, G_loss_val: {:.5f},' \
                    'l1: {:5f}, ms-ssim: {:5f}, gradient_penalty: {:5f}, lr: {:.5f}'.format(
                        epoch, i, D_loss.item(), D_loss_real.item(), D_loss_fake.item(), D_loss_real_val.item(),
                        D_loss_fake_val.item(), G_loss.item(), G_loss_val.item(), l1_loss.item(), msssim_loss.item(),
                        gradient_penalty.item(), g_sch.get_last_lr()[0]))
                writer.add_scalar("loss/D_loss", D_loss.item(), tot_iter)
                writer.add_scalar("loss/D_loss_real", D_loss_real.item(), tot_iter)
                writer.add_scalar("loss/D_loss_fake", D_loss_fake.item(), tot_iter)
                # writer.add_scalar("loss/D_loss_real_val", D_loss_real_val.item(), tot_iter)
                # writer.add_scalar("loss/D_loss_fake_val", D_loss_fake_val.item(), tot_iter)
                writer.add_scalar("loss/G_loss", G_loss.item(), tot_iter)
                writer.add_scalar("loss/G_loss_val", G_loss_val.item(), tot_iter)
                writer.add_scalar("loss/l1", l1_loss.item(), tot_iter)
                writer.add_scalar("loss/ms-ssim", msssim_loss.item(), tot_iter)
                writer.add_scalar("loss/gradient_penalty", gradient_penalty.item(), tot_iter)
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
            image = G_out.clone().detach()
            image = batch_image_merge(image)
            image = imagetensor2np(image)
            mask = batch_image_merge(mask)
            mask = imagetensor2np(mask)
            G_out = G_out / 2 + 0.5
            G_out = G_out.clamp(0, 1)
            save_image(G_out, os.path.join(root, "logs/output-{}.png".format(epoch)))
            writer.add_image('image{}/mask'.format(epoch), mask, tot_iter, dataformats='HWC')
            writer.add_image('image{}/fake'.format(epoch), cv2.cvtColor(image, cv2.COLOR_BGR2RGB), tot_iter,
                             dataformats='HWC')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--test", default=False, action='store_true')
    args = parser.parse_args()
    train(open_config(args.root), args.root)
